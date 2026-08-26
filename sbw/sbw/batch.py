"""GPU-accelerated watermark batch processing with Philox RNG.

This module implements the core SBW (Stateless Bernoulli Watermarking) algorithm
with GPU-fused operations for efficient batch inference.
"""

from __future__ import annotations
import torch

from .prf import seeding_scheme_lookup, prf_lookup, extend_context_for_selfsalt


# ---------------------------------------------------------------------------
# Philox 4x32-10 counter-based RNG (vectorized, pure PyTorch)
# Matches PyTorch's PhiloxRNGEngine exactly (aten/src/ATen/core/PhiloxRNGEngine.h)
# ---------------------------------------------------------------------------
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85


def _philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Full Philox 4x32-10: 10 rounds with key bumping.

    All inputs are int64 tensors holding uint32 values (0..2^32-1).
    Returns all four output words (c0, c1, c2, c3).
    """
    for i in range(10):
        # Round function (matches PyTorch's single_round exactly)
        prod0 = c0 * _PHILOX_M0
        prod2 = c2 * _PHILOX_M1
        hi0 = (prod0 >> 32) & 0xFFFFFFFF
        lo0 = prod0 & 0xFFFFFFFF
        hi1 = (prod2 >> 32) & 0xFFFFFFFF
        lo1 = prod2 & 0xFFFFFFFF
        c0 = (hi1 ^ c1 ^ k0) & 0xFFFFFFFF
        c1 = lo1
        c2 = (hi0 ^ c3 ^ k1) & 0xFFFFFFFF
        c3 = lo0
        # Key bump (after rounds 0-8, the bump after round 9 is harmless)
        k0 = (k0 + _PHILOX_W0) & 0xFFFFFFFF
        k1 = (k1 + _PHILOX_W1) & 0xFFFFFFFF
    return c0, c1, c2, c3


# Triton-friendly Philox: uses int32 tensors with unsigned interpretation.
# The key trick: cast to int64 with & 0xFFFFFFFF before multiply to avoid sign extension.
_MASK32 = 0xFFFFFFFF


def _philox4x32_10_i32(c0, c1, c2, c3, k0, k1):
    """Philox 4x32-10 using int32 tensors. Triton-fusible.

    All inputs/outputs are int32 tensors. Uses & 0xFFFFFFFF on int64 intermediates
    to ensure unsigned multiply semantics. Produces identical results to _philox4x32_10.
    """
    for _ in range(10):
        # Unsigned multiply: cast to int64, mask to get unsigned value, multiply
        prod0 = (c0.long() & _MASK32) * _PHILOX_M0
        prod2 = (c2.long() & _MASK32) * _PHILOX_M1
        hi0 = (prod0 >> 32).to(torch.int32)
        lo0 = prod0.to(torch.int32)
        hi1 = (prod2 >> 32).to(torch.int32)
        lo1 = prod2.to(torch.int32)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        # Key bump (wraps in int32 naturally)
        k0 = (k0.long() + _PHILOX_W0).to(torch.int32)
        k1 = (k1.long() + _PHILOX_W1).to(torch.int32)
    return c0, c1, c2, c3


# Compile into a single fused CUDA kernel
_philox4x32_10_compiled = torch.compile(_philox4x32_10, mode="max-autotune")

# PyTorch's uint32_to_uniform_float: (value & 0x7FFFFFFF) * scale
_UNIFORM_SCALE = 4.6566127342e-10  # 1.0 / 2^31


def _philox_setup(seeds: torch.LongTensor, n_calls: int):
    """Setup Philox counter and key tensors from seeds.

    Args:
        seeds: (B,) or (B, N) seed tensor
        n_calls: number of Philox calls needed

    Returns:
        (c0, c1, zeros, k0, k1) ready for _philox4x32_10
    """
    B = seeds.shape[0]
    device = seeds.device

    # Key from seed (matches PyTorch: key[0] = seed low, key[1] = seed high)
    k0 = (seeds & 0xFFFFFFFF).unsqueeze(-1).expand(*seeds.shape, n_calls)
    k1 = ((seeds >> 32) & 0xFFFFFFFF).unsqueeze(-1).expand(*seeds.shape, n_calls)

    # Counter: offset in (c0, c1), subsequence=0 in (c2, c3)
    offsets = torch.arange(n_calls, device=device, dtype=torch.int64)
    c0 = (offsets & 0xFFFFFFFF).expand(B, -1)
    c1 = ((offsets >> 32) & 0xFFFFFFFF).expand(B, -1)
    zeros = torch.zeros(B, n_calls, device=device, dtype=torch.int64)

    return c0, c1, zeros, k0, k1


# ---------------------------------------------------------------------------
# UNUSED IN PRODUCTION - Kept as reference implementation for testing.
# Tests use this to: (1) verify bit-exactness with PyTorch's Philox RNG,
# (2) validate that fused kernels (philox_apply_watermark, philox_at_positions)
# produce equivalent results.
# ---------------------------------------------------------------------------
def philox_uniform_bxv(seeds: torch.LongTensor, vocab_size: int) -> torch.FloatTensor:
    """Generate a (B, V) matrix of uniform [0,1) floats, one seed per row.

    Bit-identical to torch.rand() with the same seed on CPU.
    Uses Philox 4x32-10 with subsequence=0, offset=0,1,2,...
    Each Philox call produces 4 random floats (output words 0-3).
    """
    B = seeds.shape[0]
    n_calls = (vocab_size + 3) // 4
    c0, c1, zeros, k0, k1 = _philox_setup(seeds, n_calls)

    r0, r1, r2, r3 = _philox4x32_10_compiled(
        c0, c1, zeros, zeros.clone(), k0, k1
    )

    # Interleave 4 outputs per call: [r0_0, r1_0, r2_0, r3_0, r0_1, ...]
    # Stack to (B, n_calls, 4) then reshape to (B, n_calls*4)
    all_outputs = torch.stack([r0, r1, r2, r3], dim=2).reshape(B, n_calls * 4)

    # Trim to vocab_size
    all_outputs = all_outputs[:, :vocab_size]

    # PyTorch's float conversion: (value & 0x7FFFFFFF) * (1/2^31)
    return (all_outputs & 0x7FFFFFFF).float() * _UNIFORM_SCALE


def _philox_apply_watermark_tensop(c0, c1, c2, c3, k0, k1, logits, int_thresholds, delta_vec):
    """Fused Philox 4x32-10 → integer threshold → bias addition.

    Runs Philox, checks (output & 0x7FFFFFFF) < int_thresholds for each of the
    4 output words, and adds delta to the corresponding logits positions.
    int_thresholds: (B, 1) per-sequence thresholds.
    delta_vec: (B, 1) per-sequence delta values.
    """
    c0, c1, c2, c3 = _philox4x32_10(c0, c1, c2, c3, k0, k1)

    # Interleave, threshold, and add bias in one expression
    all_r = torch.stack([c0, c1, c2, c3], dim=2)
    B, n_calls, _ = all_r.shape
    all_r = all_r.reshape(B, n_calls * 4)[:, :logits.shape[1]]
    bias = ((all_r & 0x7FFFFFFF) < int_thresholds).to(logits.dtype) * delta_vec
    return logits + bias


_philox_apply_watermark_compiled = torch.compile(
    _philox_apply_watermark_tensop, mode="max-autotune"
)


def philox_apply_watermark(seeds: torch.LongTensor, logits: torch.FloatTensor,
                           gamma, delta) -> torch.FloatTensor:
    """Fused Philox greenlist + bias: single kernel, no intermediate tensors.

    gamma/delta: scalar or (B,) tensor for per-sequence values.
    """
    B = seeds.shape[0]
    vocab_size = logits.shape[1]
    device = seeds.device
    n_calls = (vocab_size + 3) // 4

    # Convert to (B, 1) tensors for broadcasting
    if isinstance(gamma, (int, float)):
        int_thresholds = torch.full((B, 1), int(gamma * 0x7FFFFFFF), device=device, dtype=torch.int64)
    else:
        int_thresholds = (gamma.to(device).double() * 0x7FFFFFFF).long().unsqueeze(1)

    if isinstance(delta, (int, float)):
        delta_vec = torch.full((B, 1), delta, device=device, dtype=logits.dtype)
    else:
        delta_vec = delta.to(device=device, dtype=logits.dtype).unsqueeze(1)

    c0, c1, zeros, k0, k1 = _philox_setup(seeds, n_calls)

    return _philox_apply_watermark_compiled(
        c0, c1, zeros, zeros.clone(), k0, k1, logits, int_thresholds, delta_vec
    )


# ---------------------------------------------------------------------------
# GPU-native hash and Philox utilities for self-salt
# ---------------------------------------------------------------------------

def hashint_gpu(x: torch.LongTensor) -> torch.LongTensor:
    """Bob Jenkins integer hash, GPU-native. +1 input to avoid hash(0)=0."""
    i = (x + 1).to(torch.int32)
    i = i - (i << 6)
    i = i ^ (i >> 17)
    i = i - (i << 9)
    i = i ^ (i << 4)
    i = i - (i << 3)
    i = i ^ (i << 10)
    i = i ^ (i >> 15)
    return i.to(torch.long)


def _selfsalt_fused_inner(context, logits, gamma_int, delta, hash_key, num_candidates):
    """Top-k selfhash: topk → anchored_minhash PRF → Philox → threshold → bias.

    Returns (bias, candidates) in k-space. Caller does scatter_add_ outside.

    Design choices:
    - NO CUDAGraphs: scatter_add_ on the input would disable CUDAGraphs anyway,
      and returning (B,K) tensors avoids the (B,V) clone+scatter that torch.compile
      would otherwise generate as two extra Triton memcpy kernels.
    - topk INSIDE compiled graph: allows Triton to fuse topk output directly into
      the Philox computation without materializing intermediate (B,K) tensors.
    - scatter OUTSIDE: eliminates the (B,V) clone that torch.compile needs when
      scatter_add_ is inside the graph (it generates copy-in + copy-out kernels).
      The external scatter_add_ on K=100 values is negligible (~0.01ms).
    - int32 Philox: avoids Triton "scalar out of range" errors from uint32 constants.
      Uses (x.long() & 0xFFFFFFFF) * M pattern for unsigned multiply.
    """
    B = logits.shape[0]
    V = logits.shape[1]
    h = context.shape[1]

    # Top-k inside compiled graph
    if num_candidates < V:
        _, candidates = logits.topk(num_candidates, dim=-1)  # (B, N)
    else:
        candidates = torch.arange(V, device=logits.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    N = candidates.shape[1]
    
    # --- Inline anchored_minhash_prf ---
    prefix = context[:, 1:]  # (B, h-1) - drop first token
    h_prefix = hashint_gpu(prefix)
    h_anchor = hashint_gpu(candidates)

    prefix_term = h_prefix.unsqueeze(2) * h_anchor.unsqueeze(1)  # (B, h-1, N)
    cand_term = h_anchor * h_anchor  # (B, N)
    
    prefix_min = (hash_key * prefix_term).min(dim=1).values  # (B, N)
    cand_val = hash_key * cand_term  # (B, N)
    seeds = torch.minimum(prefix_min, cand_val)  # (B, N)
    
    # --- Inline Philox with integer threshold ---
    call_idx = (candidates // 4).to(torch.int32)
    word_idx = (candidates % 4).to(torch.int32)
    
    k0 = (seeds & 0xFFFFFFFF).to(torch.int32)
    k1 = ((seeds >> 32) & 0xFFFFFFFF).to(torch.int32)
    zeros = torch.zeros_like(call_idx)
    
    r0, r1, r2, r3 = _philox4x32_10_i32(call_idx, zeros, zeros.clone(), zeros.clone(), k0, k1)
    
    selected = torch.where(word_idx == 0, r0,
               torch.where(word_idx == 1, r1,
               torch.where(word_idx == 2, r2, r3)))
    
    is_green = (selected & 0x7FFFFFFF) < gamma_int  # (B, N) bool
    
    # --- return bias + candidates (scatter done outside compiled graph) ---
    bias = (is_green.to(logits.dtype)) * delta
    return bias, candidates


_selfsalt_fused_compiled = torch.compile(_selfsalt_fused_inner, mode="max-autotune")


def _selfsalt_fullvocab_inner(context, logits, gamma_int, delta, hash_key, num_candidates):
    """Fullvocab selfhash: PRF → Philox → threshold → add bias.

    Returns logits (mutated in-place). No copy-back needed in logits processor.

    Design choices:
    - IN-PLACE mutation (logits += bias): eliminates (B,V) tensor allocation and
      copy-back. Disables CUDAGraphs, but at (B, 151K) tensor sizes the memory
      bandwidth savings far outweigh the ~50µs CUDAGraph launch benefit.
      Measured 2-3x faster than the new-tensor + copy-back approach at all B.
    - NO clone: mutates the input directly. The logits processor owns the tensor
      and uses the result immediately — no aliasing concerns.
    - NO topk: fullvocab computes Philox at all V positions using arange-based indexing.
      Avoids the topk sort cost entirely. At B≤32 this is faster than SynthID's
      topk+reweight even though we process 1500x more tokens.
    - Position-based Philox: uses positions//4 and positions%4 pattern (same as topk
      path) rather than the old stack+reshape approach, avoiding a (B, V/4, 4)
      intermediate tensor that was 2x slower.
    - Broadcast optimization: hashint_gpu computed on (V,) positions tensor instead of
      (B, V) candidates, then broadcast. Saves 40% at B=512 by avoiding redundant hash
      computation across batch dimension.
    """
    B = logits.shape[0]
    V = logits.shape[1]

    # Use (V,) positions instead of (B, V) candidates — hash once, broadcast to all B
    positions = torch.arange(V, device=logits.device, dtype=torch.long)
    h_pos = hashint_gpu(positions)  # (V,) — computed once

    prefix = context[:, 1:]
    h_prefix = hashint_gpu(prefix)  # (B, H-1)

    # PRF: anchored_minhash with broadcast (V,) → (B, V)
    prefix_term = h_prefix.unsqueeze(2) * h_pos.unsqueeze(0).unsqueeze(0)  # (B, H-1, V)
    prefix_min = (hash_key * prefix_term).min(dim=1).values  # (B, V)
    cand_val = hash_key * h_pos * h_pos  # (V,) — broadcasts to (B, V)
    seeds = torch.minimum(prefix_min, cand_val)  # (B, V)

    # Philox setup from positions (V,) — broadcast where needed
    call_idx = (positions // 4).to(torch.int32)  # (V,)
    word_idx = (positions % 4).to(torch.int32)   # (V,)

    k0 = (seeds & 0xFFFFFFFF).to(torch.int32)
    k1 = ((seeds >> 32) & 0xFFFFFFFF).to(torch.int32)
    zeros = torch.zeros(B, V, device=logits.device, dtype=torch.int32)

    r0, r1, r2, r3 = _philox4x32_10_i32(
        call_idx.unsqueeze(0).expand(B, -1),
        zeros, zeros.clone(), zeros.clone(), k0, k1
    )

    selected = torch.where(word_idx == 0, r0,
               torch.where(word_idx == 1, r1,
               torch.where(word_idx == 2, r2, r3)))

    is_green = (selected & 0x7FFFFFFF) < gamma_int
    logits += (is_green.to(logits.dtype)) * delta
    return logits


_selfsalt_fullvocab_compiled = torch.compile(_selfsalt_fullvocab_inner, mode="max-autotune")


def _selfsalt_fullvocab_clone_inner(context, logits, gamma_int, delta, hash_key, N):
    """Same as _selfsalt_fullvocab_inner but clones logits instead of in-place mutation.

    CUDAGraph-compatible: does not mutate the input tensor.
    """
    B = logits.shape[0]
    V = logits.shape[1]

    positions = torch.arange(V, device=logits.device, dtype=torch.long)
    h_pos = hashint_gpu(positions)

    prefix = context[:, 1:]
    h_prefix = hashint_gpu(prefix)

    prefix_term = h_prefix.unsqueeze(2) * h_pos.unsqueeze(0).unsqueeze(0)
    prefix_min = (hash_key * prefix_term).min(dim=1).values
    cand_val = hash_key * h_pos * h_pos
    seeds = torch.minimum(prefix_min, cand_val)

    call_idx = (positions // 4).to(torch.int32)
    word_idx = (positions % 4).to(torch.int32)

    k0 = (seeds & 0xFFFFFFFF).to(torch.int32)
    k1 = ((seeds >> 32) & 0xFFFFFFFF).to(torch.int32)
    zeros = torch.zeros(B, V, device=logits.device, dtype=torch.int32)

    r0, r1, r2, r3 = _philox4x32_10_i32(
        call_idx.unsqueeze(0).expand(B, -1),
        zeros, zeros.clone(), zeros.clone(), k0, k1
    )

    selected = torch.where(word_idx == 0, r0,
               torch.where(word_idx == 1, r1,
               torch.where(word_idx == 2, r2, r3)))

    is_green = (selected & 0x7FFFFFFF) < gamma_int
    return logits + (is_green.to(logits.dtype)) * delta


_selfsalt_fullvocab_clone_compiled = torch.compile(_selfsalt_fullvocab_clone_inner, mode="max-autotune")


def _simple_fused_inner(context, logits, gamma, delta, hash_key):
    """Fullvocab simple_4: additive_prf seed → Philox over all V → threshold → add bias.

    Returns logits (mutated in-place). No copy-back needed in logits processor.

    Design choices:
    - IN-PLACE mutation (logits += bias): eliminates (B,V) tensor allocation and
      copy-back. Disables CUDAGraphs, but at (B, 151K) tensor sizes the memory
      bandwidth savings far outweigh the ~50µs CUDAGraph launch benefit.
      Measured 2-3x faster than the new-tensor + copy-back approach at all B.
    - NO topk: biases the entire vocabulary. Faster than topk variants at B≤20 because
      it avoids the O(V·log K) partial sort. The Philox over 151K tokens is cheaper
      than sorting 151K tokens to find top-100.
    - Position-based Philox: computes Philox at positions 0..V-1 using arange//4 and
      arange%4, then selects the correct output via torch.where. Avoids the old
      stack([r0,r1,r2,r3]).reshape() pattern that materialized a (B, V/4, 4) tensor.
    - Single seed per batch element: additive_prf produces one scalar seed per row
      (hash_key * sum(context)). The seed is broadcast to all V positions via expand.
      This is simpler and faster than selfhash which computes per-token seeds.
    """
    B = logits.shape[0]
    V = logits.shape[1]

    # Inline additive_prf
    seeds = hash_key * context.sum(dim=1)  # (B,)

    # Position-based Philox (same pattern as selfhash — avoids stack+reshape)
    positions = torch.arange(V, device=logits.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    call_idx = (positions // 4).to(torch.int32)
    word_idx = (positions % 4).to(torch.int32)

    k0 = (seeds & 0xFFFFFFFF).to(torch.int32).unsqueeze(1).expand(-1, V)
    k1 = ((seeds >> 32) & 0xFFFFFFFF).to(torch.int32).unsqueeze(1).expand(-1, V)
    zeros = torch.zeros_like(call_idx)

    r0, r1, r2, r3 = _philox4x32_10_i32(call_idx, zeros, zeros.clone(), zeros.clone(), k0, k1)

    selected = torch.where(word_idx == 0, r0,
               torch.where(word_idx == 1, r1,
               torch.where(word_idx == 2, r2, r3)))

    gamma_int = int(gamma * 0x7FFFFFFF) if isinstance(gamma, (int, float)) else (gamma.double() * 0x7FFFFFFF).to(torch.int32)
    is_green = (selected & 0x7FFFFFFF) < gamma_int
    logits += is_green.to(logits.dtype) * delta
    return logits


_simple_fused_compiled = torch.compile(_simple_fused_inner, mode="max-autotune")


def _simple_topk_fused_inner(context, logits, gamma_int, delta, hash_key, num_candidates):
    """Top-k simple_4: topk → additive_prf seed → Philox at K positions → threshold → bias.

    Returns (bias, candidates) in k-space. Caller does scatter_add_ outside.

    Design choices:
    - NO CUDAGraphs: same rationale as _selfsalt_fused_inner — returning k-space
      tensors and scattering outside eliminates two (B,V) memcpy Triton kernels
      that torch.compile generates for in-graph scatter_add_.
    - topk INSIDE compiled graph: Triton can fuse the topk output buffer directly
      into subsequent int32 operations without an extra kernel launch.
    - scatter OUTSIDE: at K=100, scatter_add_ writes 100 values into a 151K tensor.
      Cost is ~0.01ms — negligible vs the 0.75ms saved by eliminating the clone.
    - Crossover with SynthID at B≈20: below this, SBW is faster (simpler PRF,
      less compute). Above this, SynthID wins because its float32 reweighting
      fuses perfectly with topk while our int32 Philox creates type-transition
      barriers that prevent full fusion.
    """
    N = num_candidates

    # Top-k inside compiled graph (shape is fixed per scheme)
    _, candidates = logits.topk(N, dim=-1)  # (B, N)

    # Inline additive_prf
    seeds = hash_key * context.sum(dim=1)  # (B,)

    # Philox at candidate positions (int32 path for Triton fusion)
    call_idx = (candidates // 4).to(torch.int32)
    word_idx = (candidates % 4).to(torch.int32)
    k0 = (seeds & 0xFFFFFFFF).to(torch.int32).unsqueeze(1).expand(-1, N)
    k1 = ((seeds >> 32) & 0xFFFFFFFF).to(torch.int32).unsqueeze(1).expand(-1, N)
    zeros = torch.zeros_like(call_idx)

    r0, r1, r2, r3 = _philox4x32_10_i32(call_idx, zeros, zeros.clone(), zeros.clone(), k0, k1)

    selected = torch.where(word_idx == 0, r0,
               torch.where(word_idx == 1, r1,
               torch.where(word_idx == 2, r2, r3)))

    # Threshold and compute bias in k-space
    is_green = (selected & 0x7FFFFFFF) < gamma_int
    bias = is_green.to(logits.dtype) * delta
    return bias, candidates


_simple_topk_fused_compiled = torch.compile(_simple_topk_fused_inner, mode="max-autotune")


def philox_at_positions(seeds: torch.LongTensor, positions: torch.LongTensor) -> torch.FloatTensor:
    """Compute Philox random value at specific positions.

    Args:
        seeds: (B, N) or (B,) - Philox seeds
        positions: (B, N) or (B,) - token IDs (position in Philox sequence)

    Returns:
        Uniform [0,1) floats, same shape as input
    """
    squeeze = seeds.dim() == 1
    if squeeze:
        seeds = seeds.unsqueeze(1)
        positions = positions.unsqueeze(1)

    device = seeds.device
    call_idx = positions // 4
    word_idx = positions % 4

    k0 = seeds & 0xFFFFFFFF
    k1 = (seeds >> 32) & 0xFFFFFFFF
    c0 = call_idx & 0xFFFFFFFF
    c1 = (call_idx >> 32) & 0xFFFFFFFF
    zeros = torch.zeros_like(c0)

    r0, r1, r2, r3 = _philox4x32_10_compiled(c0, c1, zeros, zeros.clone(), k0, k1)
    all_outputs = torch.stack([r0, r1, r2, r3], dim=2)
    selected = all_outputs.gather(2, word_idx.unsqueeze(2)).squeeze(2)
    result = (selected & 0x7FFFFFFF).float() * _UNIFORM_SCALE

    return result.squeeze(1) if squeeze else result


class WatermarkBatch:
    """
    Batched watermark computation.

    Handles multiple sequences with shared gamma/delta parameters.
    """

    def __init__(
        self,
        vocab: list[int],
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        hash_key: int = 15485863,
        device: torch.device = None,
        profile: bool = False,
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.hash_key = hash_key
        self.device = device
        self.profile = profile

        prf_type, context_width, self_salt, is_fused, num_candidates = seeding_scheme_lookup(seeding_scheme)
        self.prf_type = prf_type
        self.context_width = context_width
        self.self_salt = self_salt
        self.is_fused = is_fused
        self.selfsalt_num_candidates = num_candidates if num_candidates is not None else 40
        self._prf_fn = prf_lookup[prf_type]
        self._uses_gpu_hash = self._should_use_gpu_hash(seeding_scheme, prf_type)

        if profile:
            self.profile_stats = {"seed": [], "rand": [], "rejection": [], "sort": [], "total": [], "candidates_examined": []}

    def set_seeding_scheme(self, seeding_scheme, hash_key=None):
        """Update seeding scheme and re-derive all dependent state."""
        self.seeding_scheme = seeding_scheme
        if hash_key is not None:
            self.hash_key = hash_key
        prf_type, context_width, self_salt, is_fused, num_candidates = seeding_scheme_lookup(seeding_scheme)
        self.prf_type = prf_type
        self.context_width = context_width
        self.self_salt = self_salt
        self.is_fused = is_fused
        self.selfsalt_num_candidates = num_candidates if num_candidates is not None else 40
        self._prf_fn = prf_lookup[prf_type]
        self._uses_gpu_hash = self._should_use_gpu_hash(seeding_scheme, prf_type)

    @staticmethod
    def _should_use_gpu_hash(seeding_scheme: str, prf_type: str) -> bool:
        """Check if this scheme/prf combination should use GPU integer hashing."""
        # gpu-selfhash-cpuhash variants use GPU code path but CPU hash function
        if "cpuhash" in seeding_scheme:
            return False
        hashing_prfs = {"simple_skip_prf", "skipgram_prf", "anchored_skipgram_prf",
                        "minhash_prf", "anchored_minhash_prf", "minskipgram_prf", "noncomm_prf"}
        return seeding_scheme.startswith("gpu-") and prf_type in hashing_prfs

    def _compute_seeds(self, context: torch.LongTensor) -> torch.LongTensor:
        """
        Compute seeds for each sequence in the batch.

        Args:
            context: Shape (B, context_width)

        Returns:
            Seeds tensor of shape (B,)
        """
        if self._uses_gpu_hash:
            return self._prf_fn(context, self.hash_key, hashint_fn=hashint_gpu)
        return self._prf_fn(context, self.hash_key)

    def apply_watermark_fused(self, context: torch.LongTensor, logits: torch.FloatTensor,
                              gamma=None, delta=None) -> torch.FloatTensor:
        """Fused Philox + threshold + bias for non-self-salt GPU schemes.

        gamma/delta: scalar or (B,) tensor. Defaults to self.gamma/self.delta.
        """
        import time

        if self.profile:
            device = context.device
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

        seeds = self._compute_seeds(context)
        result = philox_apply_watermark(
            seeds, logits,
            gamma if gamma is not None else self.gamma,
            delta if delta is not None else self.delta,
        )

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["total"].append((time.perf_counter() - start) * 1000)

        return result

    def apply_watermark_simple_fused(self, context: torch.LongTensor, logits: torch.FloatTensor,
                                     gamma=None, delta=None) -> torch.FloatTensor:
        """Fully fused additive_prf + Philox + threshold + bias for gpu-fused-simple_4.

        Single call: seed computation + compiled Philox watermark in one path.
        gamma/delta: scalar or (B,) tensor. Defaults to self.gamma/self.delta.
        """
        gamma = gamma if gamma is not None else self.gamma
        delta = delta if delta is not None else self.delta
        return _simple_fused_compiled(context, logits, gamma, delta, self.hash_key)

    def apply_watermark_topk(self, context: torch.LongTensor, logits: torch.FloatTensor,
                             gamma=None, delta=None, num_candidates: int = None) -> torch.FloatTensor:
        """Top-k watermarking for non-self-salt GPU schemes.

        Only computes Philox at top-k positions, then scatter-adds bias.
        Much faster than full-vocab when k << V.

        gamma/delta: scalar or (B,) tensor. Defaults to self.gamma/self.delta.
        num_candidates: Number of top candidates. Defaults to self.selfsalt_num_candidates or 100.
        """
        B = context.shape[0]
        V = logits.shape[1]
        N = num_candidates or getattr(self, 'selfsalt_num_candidates', 100) or 100
        gamma = gamma if gamma is not None else self.gamma
        delta = delta if delta is not None else self.delta

        # Convert gamma to integer threshold
        if isinstance(gamma, torch.Tensor):
            gamma_int = (gamma.view(B, 1).double() * 0x7FFFFFFF).long()
        else:
            gamma_int = int(gamma * 0x7FFFFFFF)

        # Convert delta
        if isinstance(delta, torch.Tensor):
            delta = delta.view(B, 1).to(logits.dtype)

        bias, candidates = _simple_topk_fused_compiled(context, logits, gamma_int, delta, self.hash_key, N)
        logits.scatter_add_(1, candidates, bias)
        return logits

    def apply_watermark_selfsalt_direct(self, context: torch.LongTensor, logits: torch.FloatTensor,
                                        gamma=None, delta=None, num_candidates: int = None) -> torch.FloatTensor:
        """Direct top-k + seed + threshold + bias for self-salt GPU schemes.

        Avoids materializing (B, V) boolean mask by directly adding delta to green candidates.

        gamma/delta: scalar or (B,) tensor. Defaults to self.gamma/self.delta.
        num_candidates: Number of top candidates to evaluate. Use 0 or vocab_size for full vocabulary.
                        Defaults to self.selfsalt_num_candidates.
        """
        import time

        B = context.shape[0]
        V = logits.shape[1]
        if num_candidates is None:
            num_candidates = self.selfsalt_num_candidates
        N = num_candidates if num_candidates > 0 and num_candidates < V else V
        device = context.device
        gamma = gamma if gamma is not None else self.gamma
        delta = delta if delta is not None else self.delta

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

        # Select candidates: topk if N < V, else all tokens
        if N < V:
            _, candidates = logits.topk(N, dim=-1)  # (B, N)
        else:
            candidates = torch.arange(V, device=device).unsqueeze(0).expand(B, -1)  # (B, V)

        # Extend context with candidates (drops first token, appends candidate)
        extended = extend_context_for_selfsalt(context, candidates)  # (B*N, h)

        # Compute seeds using regular PRF (handles GPU hash automatically)
        seeds = self._compute_seeds(extended).reshape(B, N)  # (B, N)

        # Philox threshold check
        random_vals = philox_at_positions(seeds, candidates)  # (B, N)

        # Compute per-candidate gamma threshold
        if isinstance(gamma, torch.Tensor):
            gamma = gamma.view(B, 1)
        is_green = random_vals < gamma  # (B, N)

        # Compute per-candidate delta bias
        if isinstance(delta, torch.Tensor):
            delta_bias = delta.view(B, 1).expand_as(is_green)
        else:
            delta_bias = delta

        # Scatter-add delta to green candidates
        bias = torch.zeros_like(logits)
        bias.scatter_(1, candidates, is_green.to(logits.dtype) * delta_bias)
        result = logits + bias

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["total"].append((time.perf_counter() - start) * 1000)

        return result

    def apply_watermark_selfsalt_fused(self, context: torch.LongTensor, logits: torch.FloatTensor,
                                       gamma=None, delta=None, num_candidates: int = None) -> torch.FloatTensor:
        """Fused selfhash watermarking: topk → PRF → Philox → threshold → scatter-add.

        All operations fused into a single compiled kernel. Avoids materializing:
        - (B*N, h) extended context tensor
        - (B, N, 4) Philox output stack
        - (B, V) bias tensor

        gamma/delta: scalar or (B,) tensor. Defaults to self.gamma/self.delta.
        num_candidates: Number of top candidates to evaluate. Use 0 or vocab_size for full vocabulary.
                        Defaults to self.selfsalt_num_candidates.
        """
        import time

        B = context.shape[0]
        V = logits.shape[1]
        if num_candidates is None:
            num_candidates = self.selfsalt_num_candidates
        N = num_candidates if num_candidates > 0 and num_candidates < V else V
        device = context.device
        gamma = gamma if gamma is not None else self.gamma
        delta = delta if delta is not None else self.delta

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

        # Convert gamma to integer threshold
        if isinstance(gamma, torch.Tensor):
            gamma_int = (gamma.view(B, 1).double() * 0x7FFFFFFF).long()
        else:
            gamma_int = int(gamma * 0x7FFFFFFF)

        # Convert delta to proper shape
        if isinstance(delta, torch.Tensor):
            delta = delta.view(B, 1).to(logits.dtype)

        # Fused kernel: topk → PRF → Philox → threshold
        # Slice context to last context_width tokens to match original selfhash semantics:
        # PRF uses (context_width - 1) prefix tokens + candidate, so we need exactly context_width tokens.
        cw = self.context_width
        if context.shape[1] > cw:
            context = context[:, -cw:]

        if N >= V:
            # Fullvocab: add bias inside compiled graph
            if "cudagraphs" in self.seeding_scheme:
                result = _selfsalt_fullvocab_clone_compiled(context, logits, gamma_int, delta, self.hash_key, N)
            else:
                result = _selfsalt_fullvocab_compiled(context, logits, gamma_int, delta, self.hash_key, N)
        else:
            # Top-k: return (bias, candidates) in k-space, scatter outside
            bias, candidates = _selfsalt_fused_compiled(context, logits, gamma_int, delta, self.hash_key, N)
            logits.scatter_add_(1, candidates, bias)
            result = logits

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["total"].append((time.perf_counter() - start) * 1000)

        return result

    def get_greenlist_masks(self, context: torch.LongTensor, logits: torch.FloatTensor = None,
                            gamma=None, num_candidates: int = None) -> torch.BoolTensor:
        """
        Get greenlist mask for batch of sequences.

        Args:
            context: Shape (B, context_width)
            logits:  Shape (B, vocab_size), required when self_salt=True
            gamma:   Scalar or (B,) tensor. Defaults to self.gamma.
            num_candidates: For self-salt schemes, number of top candidates to evaluate.
                           Use 0 or vocab_size for full vocabulary.
                           Defaults to self.selfsalt_num_candidates.

        Returns:
            Boolean mask of shape (B, vocab_size) where True = green token
        """
        import time

        if self.profile:
            device = context.device
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_total = time.perf_counter()

        gamma = gamma if gamma is not None else self.gamma
        if num_candidates is None:
            num_candidates = self.selfsalt_num_candidates

        if self.seeding_scheme.startswith("gpu-"):
            if self.self_salt:
                if logits is None:
                    raise ValueError("logits required for self-salt seeding schemes")
                result = self._get_greenlist_masks_gpu_selfsalt(context, logits, num_candidates=num_candidates, gamma=gamma)
            else:
                result = self._get_greenlist_masks_gpu_standard(context, gamma=gamma)
        else:
            if self.self_salt:
                if logits is None:
                    raise ValueError("logits required for self-salt seeding schemes")
                result = self._get_greenlist_masks_selfhash(context, logits)
            else:
                result = self._get_greenlist_masks_standard(context)

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["total"].append((time.perf_counter() - start_total) * 1000)

        return result

    def _get_greenlist_masks_standard(self, context: torch.LongTensor) -> torch.BoolTensor:
        """Standard green list: seed from context, randperm to select green tokens."""
        import time

        B = context.shape[0]
        device = context.device

        if self.profile:
            start_seed = time.perf_counter()

        seeds = self._compute_seeds(context)

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["seed"].append((time.perf_counter() - start_seed) * 1000)

        greenlist_masks = torch.zeros((B, self.vocab_size), dtype=torch.bool, device=device)
        greenlist_size = int(self.vocab_size * self.gamma)

        if self.profile:
            start_rand = time.perf_counter()

        for i in range(B):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seeds[i].item()) % (2**63 - 1))
            vocab_permutation = torch.randperm(self.vocab_size, device=device, generator=generator)
            greenlist_masks[i, vocab_permutation[:greenlist_size]] = True

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["rand"].append((time.perf_counter() - start_rand) * 1000)

        return greenlist_masks

    def _get_greenlist_masks_gpu_standard(self, context: torch.LongTensor,
                                               gamma=None) -> torch.BoolTensor:
        """Parallel green list using Philox 4x32-10: single vectorized (B, V) generation."""
        import time

        B = context.shape[0]
        device = context.device
        gamma = gamma if gamma is not None else self.gamma

        if self.profile:
            start_seed = time.perf_counter()

        seeds = self._compute_seeds(context)

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["seed"].append((time.perf_counter() - start_seed) * 1000)

        if self.profile:
            start_rand = time.perf_counter()

        # Use fused kernel with dummy logits and delta=1 to get mask directly
        dummy_logits = torch.zeros((B, self.vocab_size), device=device, dtype=torch.float32)
        result = philox_apply_watermark(seeds, dummy_logits, gamma, delta=1.0)
        greenlist_masks = result.bool()

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["rand"].append((time.perf_counter() - start_rand) * 1000)

        return greenlist_masks

    def _get_greenlist_masks_gpu_selfsalt(self, context: torch.LongTensor, logits: torch.FloatTensor,
                                          num_candidates: int = None, gamma=None) -> torch.BoolTensor:
        """GPU-accelerated self-salt green list using Philox threshold."""
        B = context.shape[0]
        V = self.vocab_size
        device = context.device
        gamma = gamma if gamma is not None else self.gamma
        if num_candidates is None:
            num_candidates = self.selfsalt_num_candidates

        # Select candidates
        if num_candidates == 0 or num_candidates >= V:
            candidates = torch.arange(V, device=device).unsqueeze(0).expand(B, -1)
            N = V
        else:
            _, candidates = logits.topk(num_candidates, dim=-1)  # (B, N)
            N = num_candidates

        # Extend context with candidates (drops first token, appends candidate)
        extended = extend_context_for_selfsalt(context, candidates)  # (B*N, h)

        # Compute seeds using regular PRF (handles GPU hash automatically)
        seeds = self._compute_seeds(extended).reshape(B, N)  # (B, N)

        # Philox at candidate positions
        random_vals = philox_at_positions(seeds, candidates)  # (B, N)

        # Threshold check
        if isinstance(gamma, torch.Tensor):
            gamma = gamma.view(B, 1)
        is_green = random_vals < gamma  # (B, N)

        # Scatter into full mask
        mask = torch.zeros(B, V, dtype=torch.bool, device=device)
        mask.scatter_(1, candidates, is_green)
        return mask

    def _get_greenlist_masks_selfhash(self, context: torch.LongTensor, logits: torch.FloatTensor,
                                      tail_rule: str = "fixed_compute") -> torch.BoolTensor:
        """Rejection sampling (Algorithm 3) for self-salt schemes."""
        import time

        B = context.shape[0]
        device = logits.device
        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist_masks = torch.zeros((B, self.vocab_size), dtype=torch.bool, device=device)
        total_candidates = 0

        if self.profile:
            start_rejection = time.perf_counter()

        for i in range(B):
            if self.profile:
                start_sort = time.perf_counter()

            sorted_scores, sorted_indices = logits[i].sort(descending=True)

            if self.profile:
                self.profile_stats["sort"].append((time.perf_counter() - start_sort) * 1000)

            accepted = greenlist_masks[i]

            for idx in range(len(sorted_indices)):
                candidate = sorted_indices[idx]
                # For self-salt: use last (context_width - 1) tokens + candidate = context_width total
                prefix = context[i, -(self.context_width - 1):] if self.context_width > 1 else context[i, :0]
                extended = torch.cat([prefix, candidate.unsqueeze(0)]).unsqueeze(0)
                seed = self._compute_seeds(extended)[0]
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed.item()) % (2**63 - 1))
                perm = torch.randperm(self.vocab_size, device=device, generator=generator)
                if candidate in perm[:greenlist_size]:
                    accepted[candidate] = True

                if tail_rule == "fixed_compute" and idx >= 40:
                    break
                elif tail_rule == "fixed_list_length" and accepted.sum() >= 10:
                    break
                elif tail_rule == "fixed_score":
                    next_idx = min(idx + 1, len(sorted_scores) - 1)
                    if sorted_scores[0] - sorted_scores[next_idx] > self.delta:
                        break

            total_candidates += idx + 1

        if self.profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.profile_stats["rejection"].append((time.perf_counter() - start_rejection) * 1000)
            self.profile_stats["candidates_examined"].append(total_candidates)

        return greenlist_masks

    def get_profile_summary(self):
        """Get profiling statistics."""
        if not self.profile:
            return "Profiling not enabled"

        import statistics

        summary = {}
        for component, times in self.profile_stats.items():
            if times:
                summary[component] = {
                    "mean_ms": statistics.mean(times),
                    "median_ms": statistics.median(times),
                    "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                    "total_ms": sum(times),
                    "count": len(times)
                }

        return summary
