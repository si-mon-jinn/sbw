"""vLLM logits processor for SBW watermarking.

This module provides integration with vLLM's V1 engine for real-time
watermark injection during text generation.
"""
from typing import Optional, List
import torch
from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate, MoveDirectionality
from vllm.config import VllmConfig

from sbw.batch import WatermarkBatch


class SBWLogitsProcessor(LogitsProcessor):
    """Logits processor for SBW watermarking."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
        profile: bool = False,
    ):
        try:
            super().__init__(vllm_config, device, is_pin_memory)
        except NotImplementedError:
            pass

        self.vllm_config = vllm_config
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.profile = profile

        # Live references: prompt (list[int] | None) and output (list[int], mutated by vLLM)
        self.batch_prompts: List[Optional[list]] = []
        self.batch_outputs: List[Optional[list]] = []
        self._gammas: List[Optional[float]] = []
        self._deltas: List[Optional[float]] = []
        
        # Use tokenizer vocab size (actual tokens) not model config (padded for GPU efficiency)
        tokenizer = AutoTokenizer.from_pretrained(vllm_config.model_config.model)
        self.vocab_size = len(tokenizer)

        # Single watermark instance for entire batch
        self.watermark = WatermarkBatch(
            vocab=[0] * self.vocab_size,
            gamma=0.5,
            delta=2.0,
            seeding_scheme="gpu-fused-selfhash-fullvocab",
            device=device,
            profile=profile
        )

        # Pre-allocated pinned CPU buffer and persistent GPU tensor
        self._max_batch = 256
        self._context_cpu = torch.zeros(
            self._max_batch, self.watermark.context_width,
            dtype=torch.long, pin_memory=is_pin_memory and device.type == "cuda"
        )
        self._context_gpu = torch.zeros(
            self._max_batch, self.watermark.context_width,
            dtype=torch.long, device=device
        )

        # Pre-allocated gamma/delta GPU tensors (avoid per-call allocation)
        self._gamma_gpu = torch.full((self._max_batch,), 0.5, device=device)
        self._delta_gpu = torch.full((self._max_batch,), 2.0, device=device)
        self._params_dirty = True  # rebuild GPU tensors on next apply()

        if profile:
            self.profile_stats = {"collect_tokens": [], "apply_bias": []}

    def _ensure_capacity(self, B: int):
        """Grow pre-allocated buffers if batch exceeds current capacity or context_width changed."""
        cw = self.watermark.context_width
        if B <= self._max_batch and self._context_cpu.shape[1] == cw:
            return
        self._max_batch = max(self._max_batch, B * 2)
        self._context_cpu = torch.zeros(
            self._max_batch, cw, dtype=torch.long,
            pin_memory=self.is_pin_memory and self.device.type == "cuda"
        )
        self._context_gpu = torch.zeros(
            self._max_batch, cw, dtype=torch.long, device=self.device
        )
        self._gamma_gpu = torch.zeros(self._max_batch, device=self.device)
        self._delta_gpu = torch.zeros(self._max_batch, device=self.device)
        self._params_dirty = True

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply watermark to logits for entire batch."""
        import time

        B = len(self.batch_prompts)

        if B == 0 or not any(self._deltas[:B]):
            return logits

        if self.profile:
            start_collect = time.perf_counter()

        cw = self.watermark.context_width
        self._ensure_capacity(B)
        buf = self._context_cpu

        # Fill CPU buffer directly — no Python list concatenation
        for i in range(B):
            prompt = self.batch_prompts[i]
            output = self.batch_outputs[i]
            out_len = len(output)
            if out_len >= cw:
                # All context comes from output tail
                for j in range(cw):
                    buf[i, j] = output[out_len - cw + j]
            else:
                # Need some prompt tokens to fill
                prompt_len = len(prompt) if prompt else 0
                need = cw - out_len
                if prompt_len >= need:
                    for j in range(need):
                        buf[i, j] = prompt[prompt_len - need + j]
                else:
                    # Zero-pad + whatever prompt we have
                    pad = need - prompt_len
                    for j in range(pad):
                        buf[i, j] = 0
                    for j in range(prompt_len):
                        buf[i, pad + j] = prompt[j]
                for j in range(out_len):
                    buf[i, need + j] = output[j]

        # Async copy from pinned CPU → GPU (non-blocking)
        self._context_gpu[:B].copy_(buf[:B], non_blocking=True)
        context = self._context_gpu[:B]

        if self.profile:
            self.profile_stats["collect_tokens"].append((time.perf_counter() - start_collect) * 1000)

        # Rebuild gamma/delta GPU tensors if batch changed
        if self._params_dirty:
            self._gamma_gpu[:B] = torch.tensor(self._gammas[:B], dtype=self._gamma_gpu.dtype)
            self._delta_gpu[:B] = torch.tensor(self._deltas[:B], dtype=self._delta_gpu.dtype)
            self._params_dirty = False

        gamma_vec = self._gamma_gpu[:B]
        delta_vec = self._delta_gpu[:B].to(logits.dtype)

        # Slice logits to actual vocab size (vLLM pads for GPU efficiency)
        V = self.vocab_size
        logits_v = logits[:, :V]

        if self.watermark.seeding_scheme.startswith("gpu-"):
            if self.watermark.self_salt:
                if self.watermark.is_fused:
                    # Fully fused path: no intermediate tensor materialization
                    logits[:, :V] = self.watermark.apply_watermark_selfsalt_fused(context, logits_v, gamma_vec, delta_vec)
                else:
                    # Direct self-salt: top-k + seed + threshold + bias
                    logits[:, :V] = self.watermark.apply_watermark_selfsalt_direct(context, logits_v, gamma_vec, delta_vec)
            else:
                if self.watermark.is_fused:
                    if self.watermark.selfsalt_num_candidates and self.watermark.selfsalt_num_candidates > 0:
                        # Top-k fused path: seed + Philox at k positions + scatter
                        logits[:, :V] = self.watermark.apply_watermark_topk(context, logits_v, gamma_vec, delta_vec)
                    else:
                        # Full-vocab fused path: seed + Philox over all V
                        logits[:, :V] = self.watermark.apply_watermark_simple_fused(context, logits_v, gamma_vec, delta_vec)
                else:
                    # Fused Philox + threshold + bias (seed computed separately)
                    logits[:, :V] = self.watermark.apply_watermark_fused(context, logits_v, gamma_vec, delta_vec)

            if self.profile:
                self.profile_stats["apply_bias"].append(0.0)
        else:
            # Standard CPU path: separate mask + bias
            self.watermark.gamma = self._gammas[0]
            greenlist_masks = self.watermark.get_greenlist_masks(context, logits=logits_v)

            if self.profile:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start_bias = time.perf_counter()

            logits[:, :V] = logits_v + greenlist_masks * delta_vec[:, None]

            if self.profile:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                self.profile_stats["apply_bias"].append((time.perf_counter() - start_bias) * 1000)

        return logits

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
        """Update internal state based on batch changes."""
        if not batch_update:
            return

        self._params_dirty = True

        # Handle removed sequences
        for index in batch_update.removed:
            if index < len(self.batch_prompts):
                self.batch_prompts[index] = None
                self.batch_outputs[index] = None
                self._gammas[index] = None
                self._deltas[index] = None

        # Handle added sequences
        for index, params, prompt_tokens, output_tokens in batch_update.added:
            extra = params.extra_args if params and params.extra_args else {}
            gamma = extra.get("gamma", 0.5)
            delta = extra.get("delta", 2.0)
            seeding_scheme = extra.get("seeding_scheme", "gpu-fused-selfhash-fullvocab")
            hash_key = extra.get("hash_key", 15485863)

            if seeding_scheme != self.watermark.seeding_scheme or hash_key != self.watermark.hash_key:
                self.watermark.set_seeding_scheme(seeding_scheme, hash_key=hash_key)

            # Store live references (output is mutated in-place by vLLM)
            if index < len(self.batch_prompts):
                self.batch_prompts[index] = prompt_tokens
                self.batch_outputs[index] = output_tokens
                self._gammas[index] = gamma
                self._deltas[index] = delta
            else:
                self.batch_prompts.append(prompt_tokens)
                self.batch_outputs.append(output_tokens)
                self._gammas.append(gamma)
                self._deltas.append(delta)

        # Handle moved sequences
        for idx1, idx2, move in batch_update.moved:
            if move == MoveDirectionality.UNIDIRECTIONAL:
                self.batch_prompts[idx2] = self.batch_prompts[idx1]
                self.batch_outputs[idx2] = self.batch_outputs[idx1]
                self._gammas[idx2] = self._gammas[idx1]
                self._deltas[idx2] = self._deltas[idx1]
                self.batch_prompts[idx1] = None
                self.batch_outputs[idx1] = None
                self._gammas[idx1] = None
                self._deltas[idx1] = None
            elif move == MoveDirectionality.SWAP:
                self.batch_prompts[idx1], self.batch_prompts[idx2] = \
                    self.batch_prompts[idx2], self.batch_prompts[idx1]
                self.batch_outputs[idx1], self.batch_outputs[idx2] = \
                    self.batch_outputs[idx2], self.batch_outputs[idx1]
                self._gammas[idx1], self._gammas[idx2] = \
                    self._gammas[idx2], self._gammas[idx1]
                self._deltas[idx1], self._deltas[idx2] = \
                    self._deltas[idx2], self._deltas[idx1]

        # Clean up None entries
        while None in self.batch_prompts:
            idx = self.batch_prompts.index(None)
            self.batch_prompts.pop(idx)
            self.batch_outputs.pop(idx)
            self._gammas.pop(idx)
            self._deltas.pop(idx)
