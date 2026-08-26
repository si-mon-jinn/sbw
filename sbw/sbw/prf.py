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

"""Pluggable PRF seeding schemes for SBW watermarking.

Vendored from lm-watermarking/alternative_prf_schemes.py with modifications:
- seeding_scheme_lookup returns (prf_type, context_width, self_salt, is_fused, num_candidates);
  hash_key is controlled by the caller via WatermarkBatch constructor.
- Avalanche hash variants removed (unused by any PRF function).
- Added GPU-fused scheme variants and self-salt extensions.
"""

import torch
from itertools import combinations
import re


def _parse_num_candidates(suffix: str | None, self_salt: bool) -> int | None:
    """Parse num_candidates from scheme suffix.
    
    Args:
        suffix: The suffix after the base scheme (e.g., "100", "fullvocab", or None)
        self_salt: Whether the scheme uses self-salt
        
    Returns:
        num_candidates (int) for self-salt schemes, None otherwise
        
    Raises:
        ValueError: If num_candidates specified for non-self-salt scheme
    """
    if suffix is None:
        return 40 if self_salt else None
    if not self_salt:
        raise ValueError(f"num_candidates suffix '-{suffix}' only valid for self-salt schemes")
    if suffix == "fullvocab":
        return 0
    return int(suffix)


def _parse_selfhash_scheme(scheme: str, prefix: str) -> tuple[bool, int | None]:
    """Parse selfhash scheme variants for num_candidates and cpuhash flag.
    
    Args:
        scheme: Full scheme string (e.g., "gpu-selfhash-100", "gpu-selfhash-cpuhash-fullvocab")
        prefix: The prefix to strip (e.g., "gpu-selfhash", "gpu-fused-selfhash")
        
    Returns:
        (uses_cpuhash, num_candidates)
    """
    remainder = scheme[len(prefix):]
    if not remainder:
        return False, 40
    
    # Remove leading dash
    remainder = remainder[1:] if remainder.startswith("-") else remainder
    if not remainder:
        return False, 40
    
    parts = remainder.split("-")
    uses_cpuhash = "cpuhash" in parts
    
    # Find num_candidates part
    for part in parts:
        if part == "cpuhash":
            continue
        if part == "fullvocab":
            return uses_cpuhash, 0
        if part.isdigit():
            return uses_cpuhash, int(part)
    
    return uses_cpuhash, 40


def seeding_scheme_lookup(seeding_scheme: str) -> tuple[str, int, bool, bool, int | None]:
    """Look up PRF configuration for a named seeding scheme.

    Returns:
        (prf_type, context_width, self_salt, is_fused, num_candidates)
        
        - prf_type: Name of PRF function in prf_lookup
        - context_width: Number of context tokens used for seeding
        - self_salt: Whether scheme uses self-salt (candidate token in seed)
        - is_fused: Whether scheme uses a fused GPU kernel (PRF + biasing combined)
        - num_candidates: For self-salt schemes, number of top candidates to evaluate
                         (0 = full vocabulary, None = not applicable)
    """
    if not isinstance(seeding_scheme, str):
        raise ValueError("Seeding scheme should be a string summarizing the procedure.")
    
    # --- CPU schemes ---
    if seeding_scheme in ("simple_1", "lefthash"):
        return "additive_prf", 1, False, False, None
    if seeding_scheme in ("minhash",):
        return "minhash_prf", 4, False, False, None
    if seeding_scheme in ("skipgram",):
        return "skipgram_prf", 5, False, False, None
    if seeding_scheme in ("algorithm-3", "selfhash") or seeding_scheme.startswith("selfhash-"):
        num_candidates = _parse_num_candidates(
            seeding_scheme.split("-", 1)[1] if "-" in seeding_scheme and seeding_scheme.startswith("selfhash-") else None,
            self_salt=True
        )
        return "anchored_minhash_prf", 4, True, False, num_candidates
    
    # --- CPU freeform: ff-<prf>-<cw>-<ss>[-<N>] ---
    if seeding_scheme.startswith("ff-"):
        parts = seeding_scheme.split("-")
        prf_type = parts[1]
        context_width = int(parts[2])
        self_salt = parts[3] == "True"
        suffix = parts[4] if len(parts) > 4 else None
        if prf_type not in prf_lookup:
            raise ValueError(f"Unknown prf_type '{prf_type}' in freeform scheme. Available: {list(prf_lookup.keys())}")
        num_candidates = _parse_num_candidates(suffix, self_salt)
        return prf_type, context_width, self_salt, False, num_candidates
    
    # --- GPU fused schemes: gpu-fused-selfhash[-cpuhash][-<N>|-fullvocab], gpu-fused-simple_4 ---
    if seeding_scheme.startswith("gpu-fused-"):
        if seeding_scheme == "gpu-fused-simple_4":
            return "additive_prf", 4, False, True, None
        if seeding_scheme.startswith("gpu-fused-simple_4-"):
            suffix = seeding_scheme[len("gpu-fused-simple_4-"):]
            if suffix == "fullvocab":
                return "additive_prf", 4, False, True, 0
            num_candidates = int(suffix)
            return "additive_prf", 4, False, True, num_candidates
        if seeding_scheme == "gpu-fused-selfhash-fullvocab-cudagraphs":
            return "anchored_minhash_prf", 4, True, True, 0
        if not seeding_scheme.startswith("gpu-fused-selfhash"):
            supported_fused = ["gpu-fused-simple_4", "gpu-fused-selfhash", "gpu-fused-selfhash-<N>", "gpu-fused-selfhash-fullvocab", "gpu-fused-selfhash-fullvocab-cudagraphs"]
            raise ValueError(
                f"Unsupported fused scheme '{seeding_scheme}'. "
                f"Available fused schemes: {supported_fused}"
            )
        uses_cpuhash, num_candidates = _parse_selfhash_scheme(seeding_scheme, "gpu-fused-selfhash")
        if uses_cpuhash:
            raise ValueError("gpu-fused-selfhash does not support cpuhash variant")
        return "anchored_minhash_prf", 4, True, True, num_candidates
    
    # --- GPU modular schemes ---
    if seeding_scheme in ("gpu-simple_1",):
        return "additive_prf", 1, False, False, None
    if seeding_scheme in ("gpu-minhash",):
        return "minhash_prf", 4, False, False, None
    if seeding_scheme in ("gpu-skipgram",):
        return "skipgram_prf", 5, False, False, None
    if seeding_scheme.startswith("gpu-selfhash"):
        uses_cpuhash, num_candidates = _parse_selfhash_scheme(seeding_scheme, "gpu-selfhash")
        # Note: uses_cpuhash is stored in scheme name, handled by _should_use_gpu_hash in batch.py
        return "anchored_minhash_prf", 4, True, False, num_candidates
    
    # --- GPU freeform: gpu-ff-<prf>-<cw>-<ss>[-<N>] ---
    if seeding_scheme.startswith("gpu-ff-"):
        parts = seeding_scheme.split("-")
        prf_type = parts[2]
        context_width = int(parts[3])
        self_salt = parts[4] == "True"
        suffix = parts[5] if len(parts) > 5 else None
        if prf_type not in prf_lookup:
            raise ValueError(f"Unknown prf_type '{prf_type}' in freeform scheme. Available: {list(prf_lookup.keys())}")
        num_candidates = _parse_num_candidates(suffix, self_salt)
        return prf_type, context_width, self_salt, False, num_candidates
    
    raise ValueError(f"Invalid seeding scheme '{seeding_scheme}'. Try 'simple_1'?")


# ---------------------------------------------------------------------------
# Self-salt context extension
# ---------------------------------------------------------------------------

def extend_context_for_selfsalt(context: torch.LongTensor, candidates: torch.LongTensor) -> torch.LongTensor:
    """Extend context with candidates for self-salt, keeping context_width constant.
    
    Drops the first token and appends each candidate, so the total length stays h.
    
    Args:
        context: (B, h) context tokens
        candidates: (B, N) candidate tokens to evaluate
        
    Returns:
        (B*N, h) extended contexts, one per candidate
    """
    B, h = context.shape
    N = candidates.shape[1]
    prefix = context[:, 1:]  # (B, h-1) - drop first token
    prefix_expanded = prefix.unsqueeze(2).expand(B, h - 1, N)  # (B, h-1, N)
    extended = torch.cat([prefix_expanded, candidates.unsqueeze(1)], dim=1)  # (B, h, N)
    return extended.permute(0, 2, 1).reshape(B * N, h)  # (B*N, h)


# ---------------------------------------------------------------------------
# Hash utility
# ---------------------------------------------------------------------------

# Global permutation table, generated once at import time
_rng = torch.Generator(device=torch.device("cpu"))
_rng.manual_seed(2971215073)  # fib47 is prime
_table_size = 1_000_003
_fixed_table = torch.randperm(_table_size, device=torch.device("cpu"), generator=_rng)
_gpu_tables = {}  # device -> table cache


def hashint(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Permutation-table hash. Stays on input device."""
    device = integer_tensor.device
    if device.type == "cpu":
        table = _fixed_table
    else:
        if device not in _gpu_tables:
            _gpu_tables[device] = _fixed_table.to(device)
        table = _gpu_tables[device]
    return table[integer_tensor % _table_size] + 1


# ---------------------------------------------------------------------------
# PRF functions — each maps (B, context_width) batch + salt_key → (B,) seeds
# Vectorizable PRFs use tensor ops; others loop internally.
# ---------------------------------------------------------------------------

def multiplicative_prf(context: torch.LongTensor, salt_key: int) -> torch.LongTensor:
    return salt_key * context.prod(dim=1)


def additive_prf(context: torch.LongTensor, salt_key: int) -> torch.LongTensor:
    return salt_key * context.sum(dim=1)


def minfunc_prf(context: torch.LongTensor, salt_key: int) -> torch.LongTensor:
    return salt_key * context.min(dim=1).values


def simple_skip_prf(context: torch.LongTensor, salt_key: int, k=2, hashint_fn=None) -> torch.LongTensor:
    if hashint_fn is None:
        hashint_fn = hashint
    B = context.shape[0]
    seeds = torch.empty(B, dtype=torch.long)
    for i in range(B): # TODO: this could be broadcasted if gpu int hash
        seeds[i] = hashint_fn(salt_key * context[i, ::k]).prod().item()
    return seeds


def skipgram_prf(context: torch.LongTensor, salt_key: int, hashint_fn=None) -> torch.LongTensor:
    if hashint_fn is None:
        hashint_fn = hashint
    return hashint_fn(salt_key * context[:, 0])


def anchored_skipgram_prf(context: torch.LongTensor, salt_key: int, anchor: int = -1, hashint_fn=None) -> torch.LongTensor:
    if hashint_fn is None:
        hashint_fn = hashint
    return hashint_fn(salt_key * context[:, 0]) * hashint_fn(salt_key * context[:, anchor])


def minhash_prf(context: torch.LongTensor, salt_key: int, hashint_fn=None) -> torch.LongTensor:
    if hashint_fn is None:
        hashint_fn = hashint
    return hashint_fn(salt_key * context).min(dim=1).values


def anchored_minhash_prf(context: torch.LongTensor, salt_key: int, anchor: int = -1, hashint_fn=None) -> torch.LongTensor:
    if hashint_fn is None:
        hashint_fn = hashint
    return (salt_key * hashint_fn(context) * hashint_fn(context[:, anchor:anchor+1 or None])).min(dim=1).values


def minskipgram_prf(context: torch.LongTensor, salt_key: int, k: int = 2, hashint_fn=None) -> torch.LongTensor:
    if hashint_fn is None:
        hashint_fn = hashint
    B = context.shape[0]
    seeds = torch.empty(B, dtype=torch.long)
    for i in range(B):
        skipgrams = torch.as_tensor(list(combinations(hashint_fn(salt_key * context[i]), 2)))
        seeds[i] = skipgrams.prod(dim=1).min().item()
    return seeds


def noncomm_prf(context: torch.LongTensor, salt_key: int, k: int = 2, hashint_fn=None) -> torch.LongTensor:
    if hashint_fn is None:
        hashint_fn = hashint
    B = context.shape[0]
    seeds = torch.empty(B, dtype=torch.long)
    for i in range(B):
        key = torch.as_tensor(salt_key, dtype=torch.long)
        for entry in context[i]:
            key *= hashint_fn(key * entry)
            key %= 2**32
        seeds[i] = key.item()
    return seeds


def position_prf(context: torch.LongTensor, salt_key: int, k: int = 2) -> torch.LongTensor:
    positions = torch.arange(1, context.shape[1] + 1, device=context.device)
    return (salt_key * context * positions).sum(dim=1)


prf_lookup = {
    "multiplicative_prf": multiplicative_prf,
    "additive_prf": additive_prf,
    "minfunc_prf": minfunc_prf,
    "simple_skip_prf": simple_skip_prf,
    "skipgram_prf": skipgram_prf,
    "anchored_skipgram_prf": anchored_skipgram_prf,
    "minhash_prf": minhash_prf,
    "anchored_minhash_prf": anchored_minhash_prf,
    "minskipgram_prf": minskipgram_prf,
    "noncomm_prf": noncomm_prf,
    "position_prf": position_prf,
}
