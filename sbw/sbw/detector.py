# coding=utf-8
# Derived from "A Watermark for Large Language Models" (https://arxiv.org/abs/2301.10226)
# Original code: https://github.com/jwkirchenbauer/lm-watermarking (Apache 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: Rewritten for batch detection, GPU acceleration, and
# compatibility with SBW seeding schemes.

from __future__ import annotations
import collections
import warnings
from functools import lru_cache
from math import sqrt

import scipy.stats
import torch
from tokenizers import Tokenizer

from .batch import WatermarkBatch, philox_at_positions
from .normalizers import normalization_strategy_lookup
from .utils import ngrams
from .prf import extend_context_for_selfsalt


class WatermarkDetector:
    def __init__(
        self,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        hash_key: int = 15485863,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],
        ignore_repeated_bigrams: bool = None,
        ignore_repeated_ngrams: bool = True,
        log: bool = False,
    ):
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        if ignore_repeated_bigrams is not None:
            warnings.warn(
                "ignore_repeated_bigrams is deprecated, use ignore_repeated_ngrams",
                DeprecationWarning,
                stacklevel=2,
            )
            ignore_repeated_ngrams = ignore_repeated_bigrams

        self.watermark = WatermarkBatch(
            vocab=vocab,
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            hash_key=hash_key,
            device=device,
        )

        self.gamma = gamma
        self.seeding_scheme = seeding_scheme
        self.vocab_size = len(vocab)

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.log = log
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

    def _is_token_green(self, prefix: tuple[int], target: int) -> bool:
        """Check if target is green given prefix. Works for all schemes."""
        wm = self.watermark
        context = torch.tensor(prefix, device=self.device).unsqueeze(0)  # (1, h)

        # For self_salt schemes, prefix already contains the target as last token.
        # Don't call extend_context_for_selfsalt - it would double-shift the context.

        seed = wm._compute_seeds(context)  # (1,)

        if wm.seeding_scheme.startswith("gpu-"):
            rand_val = philox_at_positions(seed, torch.tensor([target], device=self.device))
            return (rand_val < self.gamma).item()
        else:
            mask = wm._get_greenlist_masks_standard(context)
            return bool(mask[0, target])

    def _compute_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    @lru_cache(maxsize=2**16)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int) -> bool:
        """Cache-wrapped green list check."""
        return self._is_token_green(prefix, target)

    def clear_cache(self):
        """Clear the ngram score cache."""
        self._get_ngram_score_cached.cache_clear()

    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        context_width = self.watermark.context_width
        self_salt = self.watermark.self_salt
        if len(input_ids) - context_width < 1:
            raise ValueError(
                f"Must have at least 1 token to score after "
                f"the first {context_width} tokens required by the seeding scheme."
            )
        token_ngram_generator = ngrams(input_ids.cpu().tolist(), context_width + 1 - self_salt)
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        for ngram_example in frequencies_table.keys():
            prefix = ngram_example if self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(prefix, target)
        return ngram_to_watermark_lookup, frequencies_table

    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup):
        context_width = self.watermark.context_width
        self_salt = self.watermark.self_salt
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), context_width + 1 - self_salt)
        for ngram_example in ngram_examples:
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example not in used_ngrams:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
        )

    def _compute_green_mask_gpu_batched(self, input_ids: torch.Tensor):
        """Compute green mask for all positions in a single batched Philox call.
        
        Returns:
            green_mask_full: (N,) bool tensor, green status for each scored position
            green_mask_unique: (U,) bool tensor, green status for unique ngrams only
            offsets: (N,) int tensor, maps each position to its unique ngram index
        """
        h = self.watermark.context_width
        self_salt = self.watermark.self_salt
        device = self.device
        
        # Build context windows and targets
        # For self_salt: ngram is h tokens where last is target
        # For non-self_salt: ngram is h context tokens + 1 target
        if self_salt:
            # Window of h tokens, target is the last one
            ngram_len = h
            contexts = input_ids.unfold(0, h, 1)  # (N, h)
            targets = contexts[:, -1]  # (N,) last token of each window
        else:
            # h context tokens followed by target
            ngram_len = h + 1
            all_ngrams = input_ids.unfold(0, ngram_len, 1)  # (N, h+1)
            contexts = all_ngrams[:, :h]  # (N, h)
            targets = all_ngrams[:, -1]   # (N,)
        
        N = contexts.shape[0]
        
        # Handle deduplication
        if self.ignore_repeated_ngrams:
            ngrams_tensor = torch.cat([contexts, targets.unsqueeze(1)], dim=1)  # (N, ngram_len)
            unique_ngrams, inverse_indices = torch.unique(ngrams_tensor, dim=0, return_inverse=True)
            contexts_to_score = unique_ngrams[:, :h]  # (U, h)
            targets_to_score = unique_ngrams[:, -1]   # (U,)
        else:
            contexts_to_score = contexts
            targets_to_score = targets
            inverse_indices = torch.arange(N, device=device)
        
        # Compute seeds for all contexts
        seeds = self.watermark._compute_seeds(contexts_to_score)  # (U,) or (N,)
        
        # Single batched Philox call
        rand_vals = philox_at_positions(seeds, targets_to_score)  # (U,) or (N,)
        green_mask_unique = rand_vals < self.gamma  # (U,) or (N,)
        
        # Build offsets for z_at_T computation (cumulative unique index)
        if self.ignore_repeated_ngrams:
            # offsets[i] = index of first occurrence of ngram[i] in unique list
            # For z_at_T we need cumulative count of unique ngrams seen so far
            seen = {}
            offsets = []
            unique_idx = 0
            for i in range(N):
                inv_idx = inverse_indices[i].item()
                if inv_idx not in seen:
                    seen[inv_idx] = unique_idx
                    unique_idx += 1
                offsets.append(seen[inv_idx])
            offsets = torch.tensor(offsets, device=device)
            green_mask_full = green_mask_unique[inverse_indices]
        else:
            offsets = torch.arange(N, device=device)
            green_mask_full = green_mask_unique
        
        return green_mask_full, green_mask_unique, offsets

    def _score_sequence_gpu_batched(
        self,
        input_ids,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = False,
        return_p_value: bool = True,
    ):
        """Batched scoring for GPU schemes using single philox_at_positions call."""
        green_mask_full, green_mask_unique, offsets = self._compute_green_mask_gpu_batched(input_ids)
        
        num_tokens_scored = len(green_mask_unique)
        green_token_count = green_mask_unique.sum().item()
        
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict["num_tokens_scored"] = num_tokens_scored
        if return_num_green_tokens:
            score_dict["num_green_tokens"] = green_token_count
        if return_green_fraction:
            score_dict["green_fraction"] = green_token_count / num_tokens_scored
        if return_z_score:
            score_dict["z_score"] = self._compute_z_score(green_token_count, num_tokens_scored)
        if return_p_value:
            z = score_dict.get("z_score") or self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict["p_value"] = self._compute_p_value(z)
        if return_green_token_mask:
            score_dict["green_token_mask"] = green_mask_full.tolist()
        if return_z_at_T:
            sizes = torch.arange(1, len(green_mask_unique) + 1, device=green_mask_unique.device)
            seq_z_enum = torch.cumsum(green_mask_unique.float(), dim=0) - self.gamma * sizes
            seq_z_denom = torch.sqrt(sizes.float() * self.gamma * (1 - self.gamma))
            z_at_effective_T = seq_z_enum / seq_z_denom
            score_dict["z_score_at_T"] = z_at_effective_T[offsets]
        return score_dict

    def _score_sequence_legacy(
        self,
        input_ids,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = False,
        return_p_value: bool = True,
    ):
        """Original per-token scoring path for CPU schemes."""
        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(input_ids, ngram_to_watermark_lookup)

        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            green_token_count = sum(
                freq * outcome for freq, outcome in
                zip(frequencies_table.values(), ngram_to_watermark_lookup.values())
            )

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict["num_tokens_scored"] = num_tokens_scored
        if return_num_green_tokens:
            score_dict["num_green_tokens"] = green_token_count
        if return_green_fraction:
            score_dict["green_fraction"] = green_token_count / num_tokens_scored
        if return_z_score:
            score_dict["z_score"] = self._compute_z_score(green_token_count, num_tokens_scored)
        if return_p_value:
            z = score_dict.get("z_score") or self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict["p_value"] = self._compute_p_value(z)
        if return_green_token_mask:
            score_dict["green_token_mask"] = green_token_mask.tolist()
        if return_z_at_T:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_enum = torch.cumsum(green_unique.float(), dim=0) - self.gamma * sizes
            seq_z_denom = torch.sqrt(sizes.float() * self.gamma * (1 - self.gamma))
            z_at_effective_T = seq_z_enum / seq_z_denom
            score_dict["z_score_at_T"] = z_at_effective_T[offsets]
        return score_dict

    def _score_sequence(self, input_ids, **kwargs):
        """Dispatch to GPU-batched or legacy scoring based on seeding scheme."""
        if self.seeding_scheme.startswith("gpu-"):
            return self._score_sequence_gpu_batched(input_ids, **kwargs)
        return self._score_sequence_legacy(input_ids, **kwargs)

    def _compute_green_mask(self, input_ids: torch.Tensor):
        """Compute green mask using GPU-batched or legacy path based on scheme."""
        if self.seeding_scheme.startswith("gpu-"):
            return self._compute_green_mask_gpu_batched(input_ids)
        else:
            ngram_to_watermark_lookup, _ = self._score_ngrams_in_passage(input_ids)
            return self._get_green_at_T_booleans(input_ids, ngram_to_watermark_lookup)

    def _score_windows_impl_batched(self, input_ids, window_size, window_stride=1):
        """Core windowed scoring: compute max z-score across sliding windows using prefix sums."""
        green_mask, green_ids, offsets = self._compute_green_mask(input_ids)
        len_full_context = len(green_ids)

        partial_sum = torch.cumsum(green_ids.long(), dim=0)

        if window_size == "max":
            sizes = range(1, len_full_context)
        else:
            sizes = [int(x) for x in window_size.split(",") if len(x) > 0]

        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        s = window_stride

        window_fits = False
        for idx, size in enumerate(sizes):
            if size <= len_full_context:
                window_score = torch.zeros(len_full_context - size + 1, dtype=torch.long)
                window_score[0] = partial_sum[size - 1]
                window_score[1:] = partial_sum[size::s] - partial_sum[:-size:s]

                batched_z_score = (window_score - self.gamma * size) / sqrt(size * self.gamma * (1 - self.gamma))

                z_score_max_per_window[idx] = batched_z_score.max()
                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::s] = torch.maximum(cumulative_eff_z_score[size::s], z_score_at_effective_T[:-1])
                window_fits = True

        if not window_fits:
            raise ValueError(
                f"No fitting window with sizes {window_size} for context length {len_full_context}."
            )

        cumulative_z_score = cumulative_eff_z_score[offsets]
        optimal_z, optimal_idx = z_score_max_per_window.max(dim=0)
        optimal_window_size = sizes[optimal_idx]
        return optimal_z, optimal_window_size, z_score_max_per_window, cumulative_z_score, green_mask

    def _score_sequence_window(self, input_ids, window_size, window_stride=1, **kwargs):
        """Windowed scoring wrapper that formats output dict."""
        optimal_z, optimal_window_size, _, z_score_at_T, green_mask = \
            self._score_windows_impl_batched(input_ids, window_size, window_stride)

        score_dict = dict()
        if kwargs.get("return_num_tokens_scored", True):
            score_dict["num_tokens_scored"] = optimal_window_size
        if kwargs.get("return_num_green_tokens", True):
            denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma))
            score_dict["num_green_tokens"] = int(optimal_z * denom + self.gamma * optimal_window_size)
        if kwargs.get("return_green_fraction", True):
            green_count = score_dict.get("num_green_tokens",
                int(optimal_z * sqrt(optimal_window_size * self.gamma * (1 - self.gamma)) + self.gamma * optimal_window_size))
            score_dict["green_fraction"] = green_count / optimal_window_size
        if kwargs.get("return_z_score", True):
            score_dict["z_score"] = optimal_z
        if kwargs.get("return_p_value", True):
            score_dict["p_value"] = self._compute_p_value(optimal_z)
        if kwargs.get("return_z_at_T", False):
            score_dict["z_score_at_T"] = z_score_at_T
        if kwargs.get("return_green_token_mask", False):
            score_dict["green_token_mask"] = green_mask.tolist()
        return score_dict

    def dummy_detect(self, return_prediction=True, return_scores=True, **kwargs):
        """Return NaN-filled results for text too short to score."""
        output_dict = {}
        if return_scores:
            output_dict.update(dict(
                num_tokens_scored=float("nan"),
                num_green_tokens=float("nan"),
                green_fraction=float("nan"),
                z_score=float("nan"),
                p_value=float("nan"),
            ))
        if return_prediction:
            output_dict["prediction"] = False
        return output_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True

        if len(self.normalizers) > 0 and self.log:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            tokenized_text = tokenized_text["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        output_dict = {}
        if window_size is not None:
            score_dict = self._score_sequence_window(
                tokenized_text, window_size=window_size,
                window_stride=window_stride or 1, **kwargs,
            )
        else:
            score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
