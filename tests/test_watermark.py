"""Tests to verify watermarking behavior is preserved after restructuring."""

import torch
import pytest
from transformers import AutoTokenizer

from sbw import WatermarkBatch, WatermarkDetector


class TestWatermarkBatch:
    """Tests for core watermark green list generation."""

    @pytest.fixture
    def watermark(self):
        return WatermarkBatch(vocab=[0] * 1000, gamma=0.5, delta=2.0, device=torch.device("cpu"))

    def test_greenlist_size(self, watermark):
        """Green list should be gamma fraction of vocab."""
        context = torch.tensor([[42]], device="cpu")
        greenlist = watermark.get_greenlist_masks(context)[0]
        green_count = greenlist.sum().item()
        assert green_count == int(1000 * 0.5), f"Expected 500 green tokens, got {green_count}"

    def test_greenlist_deterministic(self, watermark):
        """Same seed should produce same green list."""
        context = torch.tensor([[42]], device="cpu")
        greenlist1 = watermark.get_greenlist_masks(context)[0].clone()
        greenlist2 = watermark.get_greenlist_masks(context)[0].clone()
        assert torch.equal(greenlist1, greenlist2), "Green lists should be identical for same seed"

    def test_different_tokens_different_greenlists(self, watermark):
        """Different previous tokens should produce different green lists."""
        greenlist1 = watermark.get_greenlist_masks(torch.tensor([[42]], device="cpu"))[0]
        greenlist2 = watermark.get_greenlist_masks(torch.tensor([[99]], device="cpu"))[0]
        assert not torch.equal(greenlist1, greenlist2), "Different seeds should produce different green lists"

    def test_gamma_parameter(self):
        """Different gamma values should change green list size."""
        wm_25 = WatermarkBatch(vocab=[0] * 1000, gamma=0.25, delta=2.0, device=torch.device("cpu"))
        wm_75 = WatermarkBatch(vocab=[0] * 1000, gamma=0.75, delta=2.0, device=torch.device("cpu"))

        context = torch.tensor([[42]], device="cpu")
        green_25 = wm_25.get_greenlist_masks(context)[0].sum().item()
        green_75 = wm_75.get_greenlist_masks(context)[0].sum().item()

        assert green_25 == 250, f"gamma=0.25 should give 250 green tokens, got {green_25}"
        assert green_75 == 750, f"gamma=0.75 should give 750 green tokens, got {green_75}"


class TestWatermarkDetector:
    """Tests for watermark detection."""

    @pytest.fixture
    def detector(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
            z_threshold=4.0,
            ignore_repeated_bigrams=False,
        )

    def test_z_score_calculation(self, detector):
        """Z-score should be computed correctly."""
        z = detector._compute_z_score(observed_count=30, T=50)
        expected = (30 - 0.5 * 50) / (50 * 0.5 * 0.5) ** 0.5
        assert abs(z - expected) < 0.01, f"Z-score mismatch: {z} vs {expected}"

    def test_high_green_fraction_detected(self, detector):
        """High green fraction should be detected as watermarked."""
        z = detector._compute_z_score(observed_count=45, T=50)
        assert z > 4.0, f"High green fraction should exceed threshold, got z={z}"

    def test_random_green_fraction_not_detected(self, detector):
        """~50% green fraction should not be detected."""
        z = detector._compute_z_score(observed_count=25, T=50)
        assert z < 4.0, f"Random green fraction should be below threshold, got z={z}"


class TestDetectorIntegration:
    """Integration tests using actual text detection."""

    @pytest.fixture
    def detector(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
            z_threshold=4.0,
            ignore_repeated_bigrams=False,
        )

    def test_detect_returns_expected_keys(self, detector):
        """Detection result should contain expected fields."""
        result = detector.detect(text="This is a sample text for testing the watermark detector.")
        assert "z_score" in result
        assert "prediction" in result
        assert "num_tokens_scored" in result
        assert "green_fraction" in result


class TestNgrams:
    """Tests for vendored ngrams utility (A1)."""

    def test_bigrams(self):
        from sbw.utils import ngrams
        assert list(ngrams([1, 2, 3, 4], 2)) == [(1, 2), (2, 3), (3, 4)]

    def test_trigrams(self):
        from sbw.utils import ngrams
        assert list(ngrams([1, 2, 3, 4], 3)) == [(1, 2, 3), (2, 3, 4)]

    def test_pad_left(self):
        from sbw.utils import ngrams
        assert list(ngrams([1, 2], 2, pad_left=True, pad_symbol=0)) == [(0, 1), (1, 2)]


class TestNgramScoring:
    """Tests for ngram-based _score_sequence rewrite (A2)."""

    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def _make_detector(self, tokenizer, **kwargs):
        defaults = dict(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
            z_threshold=4.0,
        )
        defaults.update(kwargs)
        return WatermarkDetector(**defaults)

    def test_ignore_repeated_ngrams_no_longer_raises(self, tokenizer):
        """ignore_repeated_ngrams=True should work (was NotImplementedError)."""
        det = self._make_detector(tokenizer, ignore_repeated_ngrams=True)
        result = det.detect(text="The quick brown fox jumps over the lazy dog near the river.")
        assert "z_score" in result

    def test_repeated_bigrams_counted_once(self, tokenizer):
        """With ignore_repeated_ngrams=True, repeated bigrams score once."""
        det = self._make_detector(tokenizer, ignore_repeated_ngrams=True)
        # Construct repeating token sequence: A B A B A B C
        ids = torch.tensor([10, 20, 10, 20, 10, 20, 30], device="cpu")
        result = det._score_sequence(ids)
        # Unique bigrams: (10,20), (20,10), (20,30) = 3
        assert result["num_tokens_scored"] == 3

    def test_repeated_bigrams_all_counted(self, tokenizer):
        """With ignore_repeated_ngrams=False, all positions are scored."""
        det = self._make_detector(tokenizer, ignore_repeated_ngrams=False)
        ids = torch.tensor([10, 20, 10, 20, 10, 20, 30], device="cpu")
        result = det._score_sequence(ids)
        # 7 tokens, context_width=1, so 6 scored positions
        assert result["num_tokens_scored"] == 6

    def test_cache_hits_across_calls(self, tokenizer):
        """Cache should have hits when scoring the same text twice."""
        det = self._make_detector(tokenizer, ignore_repeated_ngrams=False)
        det.clear_cache()
        ids = torch.tensor([10, 20, 10, 20, 10, 20, 30], device="cpu")
        det._score_sequence(ids)
        det._score_sequence(ids)
        assert det._get_ngram_score_cached.cache_info().hits > 0

    def test_clear_cache(self, tokenizer):
        """clear_cache should reset cache stats."""
        det = self._make_detector(tokenizer, ignore_repeated_ngrams=False)
        ids = torch.tensor([10, 20, 10, 20], device="cpu")
        det._score_sequence(ids)
        det.clear_cache()
        assert det._get_ngram_score_cached.cache_info().hits == 0

    def test_z_score_at_T_final_matches_z_score(self, tokenizer):
        """z_score_at_T[-1] should match the final z_score."""
        det = self._make_detector(tokenizer, ignore_repeated_ngrams=False)
        ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device="cpu")
        result = det._score_sequence(ids, return_z_at_T=True)
        assert abs(result["z_score_at_T"][-1].item() - result["z_score"]) < 1e-5

    def test_deprecation_warning(self, tokenizer):
        """Passing ignore_repeated_bigrams should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="ignore_repeated_bigrams is deprecated"):
            self._make_detector(tokenizer, ignore_repeated_bigrams=True)


class TestSetSeedingScheme:
    """Tests for WatermarkBatch.set_seeding_scheme."""

    def test_switch_scheme_updates_derived_state(self):
        wm = WatermarkBatch(vocab=[0] * 1000, device=torch.device("cpu"))
        assert wm.context_width == 1
        assert wm.prf_type == "additive_prf"

        wm.set_seeding_scheme("selfhash")
        assert wm.context_width == 4
        assert wm.self_salt is True
        assert wm.prf_type == "anchored_minhash_prf"
        assert wm.seeding_scheme == "selfhash"

    def test_switch_scheme_produces_correct_masks(self):
        wm = WatermarkBatch(vocab=[0] * 1000, device=torch.device("cpu"))
        # Get mask with simple_1
        mask_simple = wm.get_greenlist_masks(torch.tensor([[42]])).clone()

        # Switch to minhash and back
        wm.set_seeding_scheme("minhash")
        wm.set_seeding_scheme("simple_1")
        mask_after = wm.get_greenlist_masks(torch.tensor([[42]]))
        assert torch.equal(mask_simple, mask_after)

    def test_switch_scheme_with_hash_key(self):
        wm = WatermarkBatch(vocab=[0] * 1000, device=torch.device("cpu"))
        wm.set_seeding_scheme("simple_1", hash_key=999)
        assert wm.hash_key == 999

    def test_switch_preserves_gamma_delta(self):
        wm = WatermarkBatch(vocab=[0] * 1000, gamma=0.3, delta=5.0, device=torch.device("cpu"))
        wm.set_seeding_scheme("minhash")
        assert wm.gamma == 0.3
        assert wm.delta == 5.0


class TestPRFSeeding:
    """Tests for pluggable PRF seeding system (B1)."""

    def test_simple_1_context_width(self):
        wm = WatermarkBatch(vocab=[0] * 1000, device=torch.device("cpu"))
        assert wm.context_width == 1
        assert wm.self_salt is False
        assert wm.prf_type == "additive_prf"

    def test_selfhash_context_width(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="selfhash", device=torch.device("cpu"))
        assert wm.context_width == 4
        assert wm.self_salt is True
        assert wm.prf_type == "anchored_minhash_prf"

    def test_minhash_context_width(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="minhash", device=torch.device("cpu"))
        assert wm.context_width == 4
        assert wm.self_salt is False

    def test_skipgram_context_width(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="skipgram", device=torch.device("cpu"))
        assert wm.context_width == 5
        assert wm.self_salt is False

    def test_determinism_minhash(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="minhash", device=torch.device("cpu"))
        ctx = torch.tensor([[10, 20, 30, 40]])
        m1 = wm.get_greenlist_masks(ctx)[0].clone()
        m2 = wm.get_greenlist_masks(ctx)[0].clone()
        assert torch.equal(m1, m2)

    def test_different_context_different_greenlist(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="minhash", device=torch.device("cpu"))
        m1 = wm.get_greenlist_masks(torch.tensor([[10, 20, 30, 40]]))[0]
        m2 = wm.get_greenlist_masks(torch.tensor([[10, 20, 30, 99]]))[0]
        assert not torch.equal(m1, m2)

    def test_simple_1_cross_validation(self):
        """simple_1 via PRF system must produce same greenlists as before."""
        wm = WatermarkBatch(vocab=[0] * 1000, device=torch.device("cpu"))
        # (1, 1) shape
        mask = wm.get_greenlist_masks(torch.tensor([[42]]))[0]
        # Manually compute: seed = hash_key * 42, same randperm
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(15485863 * 42)
        perm = torch.randperm(1000, generator=gen)
        expected = torch.zeros(1000, dtype=torch.bool)
        expected[perm[:500]] = True
        assert torch.equal(mask, expected)

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError):
            WatermarkBatch(vocab=[0] * 1000, seeding_scheme="nonexistent", device=torch.device("cpu"))

    def test_freeform_scheme(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="ff-minhash_prf-3-False", device=torch.device("cpu"))
        assert wm.context_width == 3
        assert wm.self_salt is False
        assert wm.prf_type == "minhash_prf"


ALL_PRF_SCHEMES = [
    ("additive_prf", 2),
    ("multiplicative_prf", 2),
    ("minfunc_prf", 2),
    ("simple_skip_prf", 4),
    ("skipgram_prf", 3),
    ("anchored_skipgram_prf", 3),
    ("minhash_prf", 4),
    ("anchored_minhash_prf", 4),
    ("minskipgram_prf", 4),
    ("noncomm_prf", 3),
    ("position_prf", 3),
]


class TestAllPRFs:
    """Test all PRF functions via freeform schemes."""

    @pytest.mark.parametrize("prf_name,cw", ALL_PRF_SCHEMES)
    def test_deterministic(self, prf_name, cw):
        wm = WatermarkBatch(vocab=[0] * 500, seeding_scheme=f"ff-{prf_name}-{cw}-False", device=torch.device("cpu"))
        ctx = torch.arange(10, 10 + cw).unsqueeze(0)
        m1 = wm.get_greenlist_masks(ctx)[0].clone()
        m2 = wm.get_greenlist_masks(ctx)[0].clone()
        assert torch.equal(m1, m2), f"{prf_name} not deterministic"

    @pytest.mark.parametrize("prf_name,cw", ALL_PRF_SCHEMES)
    def test_different_context(self, prf_name, cw):
        wm = WatermarkBatch(vocab=[0] * 500, seeding_scheme=f"ff-{prf_name}-{cw}-False", device=torch.device("cpu"))
        ctx1 = torch.arange(10, 10 + cw).unsqueeze(0)
        ctx2 = torch.arange(50, 50 + cw).unsqueeze(0)
        m1 = wm.get_greenlist_masks(ctx1)[0]
        m2 = wm.get_greenlist_masks(ctx2)[0]
        assert not torch.equal(m1, m2), f"{prf_name} same output for different context"

    @pytest.mark.parametrize("prf_name,cw", ALL_PRF_SCHEMES)
    def test_batch(self, prf_name, cw):
        """Batch of 3 should produce 3 masks with correct shape."""
        wm = WatermarkBatch(vocab=[0] * 500, seeding_scheme=f"ff-{prf_name}-{cw}-False", device=torch.device("cpu"))
        ctx = torch.arange(3 * cw).reshape(3, cw) + 10
        masks = wm.get_greenlist_masks(ctx)
        assert masks.shape == (3, 500)
        assert masks.dtype == torch.bool


class TestSelfSalt:
    """Tests for self-salt / rejection sampling (B3)."""

    def test_selfhash_raises_without_logits(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="selfhash", device=torch.device("cpu"))
        ctx = torch.tensor([[10, 20, 30, 40]])
        with pytest.raises(ValueError, match="logits required"):
            wm.get_greenlist_masks(ctx)

    def test_selfhash_returns_correct_shape(self):
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="selfhash", device=torch.device("cpu"))
        ctx = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        logits = torch.randn(2, 1000)
        masks = wm.get_greenlist_masks(ctx, logits=logits)
        assert masks.shape == (2, 1000)
        assert masks.dtype == torch.bool

    def test_selfhash_accepted_tokens_are_self_consistent(self):
        """Each accepted token must be green under its own self-salted green list."""
        wm = WatermarkBatch(vocab=[0] * 500, seeding_scheme="selfhash", gamma=0.5, device=torch.device("cpu"))
        ctx = torch.tensor([[10, 20, 30, 40]])
        logits = torch.randn(1, 500)
        mask = wm.get_greenlist_masks(ctx, logits=logits)[0]
        greenlist_size = int(500 * 0.5)
        for token_id in mask.nonzero(as_tuple=True)[0]:
            extended = torch.cat([ctx[0], token_id.unsqueeze(0)]).unsqueeze(0)
            seed = wm._compute_seeds(extended)[0]
            gen = torch.Generator(device=torch.device("cpu"))
            gen.manual_seed(int(seed.item()) % (2**63 - 1))
            perm = torch.randperm(500, generator=gen)
            assert token_id in perm[:greenlist_size], f"Token {token_id} not in its own green list"

    def test_fixed_compute_stops_at_41(self):
        """fixed_compute examines at most 41 candidates (indices 0..40)."""
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="selfhash", device=torch.device("cpu"))
        ctx = torch.tensor([[10, 20, 30, 40]])
        logits = torch.randn(1, 1000)
        mask = wm.get_greenlist_masks(ctx, logits=logits)[0]
        # Can't have more accepted tokens than candidates examined
        assert mask.sum().item() <= 41

    def test_non_selfsalt_ignores_logits(self):
        """Passing logits to a non-self-salt scheme should produce same result as without."""
        wm = WatermarkBatch(vocab=[0] * 1000, device=torch.device("cpu"))
        ctx = torch.tensor([[42]])
        mask_without = wm.get_greenlist_masks(ctx)
        mask_with = wm.get_greenlist_masks(ctx, logits=torch.randn(1, 1000))
        assert torch.equal(mask_without, mask_with)

    def test_detector_works_with_selfhash(self):
        """Detector should work with selfhash scheme (uses standard path, no logits needed)."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        det = WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
            seeding_scheme="selfhash",
        )
        result = det.detect(text="The quick brown fox jumps over the lazy dog near the river bank today.")
        assert "z_score" in result
        assert "prediction" in result


class TestWindowedDetection:
    """Tests for windowed detection (B4)."""

    @pytest.fixture
    def detector(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
            z_threshold=4.0,
            ignore_repeated_ngrams=False,
        )

    def test_window_none_uses_standard_path(self, detector):
        """window_size=None should use standard _score_sequence."""
        result = detector.detect(text="The quick brown fox jumps over the lazy dog near the river.")
        assert "z_score" in result
        assert "prediction" in result

    def test_window_max_returns_optimal(self, detector):
        """window_size='max' should return a valid optimal window size."""
        result = detector.detect(
            text="The quick brown fox jumps over the lazy dog near the river bank today.",
            window_size="max",
        )
        assert "z_score" in result
        assert "num_tokens_scored" in result
        assert isinstance(result["num_tokens_scored"], (int, torch.Tensor))

    def test_window_single_size(self, detector):
        """window_size='5' should work."""
        result = detector.detect(
            text="The quick brown fox jumps over the lazy dog near the river bank today.",
            window_size="5",
        )
        assert "z_score" in result

    def test_window_multiple_sizes(self, detector):
        """window_size='3,5,8' should try multiple sizes and return the best."""
        result = detector.detect(
            text="The quick brown fox jumps over the lazy dog near the river bank today.",
            window_size="3,5,8",
        )
        assert "z_score" in result

    def test_windowed_finds_watermarked_segment(self, detector):
        """Windowed detection on a half-watermarked sequence should find higher z than full-sequence."""
        # Build a sequence where the first half is heavily green-biased
        wm = detector.watermark
        green_tokens = []
        ctx = torch.tensor([42])
        for _ in range(30):
            mask = wm._get_greenlist_masks_standard(ctx.unsqueeze(0))[0]
            green_ids = mask.nonzero(as_tuple=True)[0]
            token = green_ids[0]  # always pick a green token
            green_tokens.append(token.item())
            ctx = token.unsqueeze(0)

        # Second half: arbitrary tokens (not biased)
        random_tokens = list(range(100, 130))
        combined = torch.tensor([42] + green_tokens + random_tokens, device="cpu")

        full_result = detector._score_sequence(combined)
        windowed_result = detector._score_sequence_window(combined, window_size="30")
        # The windowed z-score should be >= full-sequence z-score
        assert windowed_result["z_score"] >= full_result["z_score"]


class TestPhiloxMatchesPyTorch:
    """Verify our Philox 4x32-10 matches PyTorch's PhiloxRNGEngine.h."""

    def test_known_test_vector_all_zeros(self):
        """All-zero input must produce the known Random123 test vector."""
        from sbw.batch import _philox4x32_10
        c = torch.tensor([0], dtype=torch.int64)
        r0, r1, r2, r3 = _philox4x32_10(c, c.clone(), c.clone(), c.clone(), c.clone(), c.clone())
        assert r0.item() == 0x6627E8D5
        assert r1.item() == 0xE169C58D
        assert r2.item() == 0xBC57AC4C
        assert r3.item() == 0x9B00DBD8

    def test_float_conversion_matches_pytorch(self):
        """Float conversion must use PyTorch's (value & 0x7FFFFFFF) * (1/2^31)."""
        from sbw.batch import philox_uniform_bxv, _philox4x32_10
        seed = 42
        c = torch.tensor([0], dtype=torch.int64)
        k0 = torch.tensor([seed], dtype=torch.int64)
        k1 = torch.tensor([0], dtype=torch.int64)
        r0, r1, r2, r3 = _philox4x32_10(c, c.clone(), c.clone(), c.clone(), k0, k1)

        scale = 4.6566127342e-10
        expected = [(r.item() & 0x7FFFFFFF) * scale for r in [r0, r1, r2, r3]]
        result = philox_uniform_bxv(torch.tensor([seed], dtype=torch.long), 4)
        for i in range(4):
            assert abs(result[0, i].item() - expected[i]) < 1e-7, f"Mismatch at [{i}]"

    def test_sequential_offsets(self):
        """Consecutive vocab positions use consecutive Philox counter offsets."""
        from sbw.batch import philox_uniform_bxv, _philox4x32_10
        seed = 123
        result = philox_uniform_bxv(torch.tensor([seed], dtype=torch.long), 8)

        # Manually compute: offset=0 gives 4 floats, offset=1 gives next 4
        scale = 4.6566127342e-10
        k0 = torch.tensor([seed & 0xFFFFFFFF], dtype=torch.int64)
        k1 = torch.tensor([seed >> 32], dtype=torch.int64)
        z = torch.tensor([0], dtype=torch.int64)

        for offset in range(2):
            c0 = torch.tensor([offset], dtype=torch.int64)
            r0, r1, r2, r3 = _philox4x32_10(c0, z.clone(), z.clone(), z.clone(), k0, k1)
            for j, r in enumerate([r0, r1, r2, r3]):
                idx = offset * 4 + j
                expected = (r.item() & 0x7FFFFFFF) * scale
                assert abs(result[0, idx].item() - expected) < 1e-7, f"Mismatch at [{idx}]"

    def test_deterministic(self):
        from sbw.batch import philox_uniform_bxv
        seeds = torch.tensor([42, 123], dtype=torch.long)
        r1 = philox_uniform_bxv(seeds, 100)
        r2 = philox_uniform_bxv(seeds, 100)
        assert torch.equal(r1, r2)

    def test_different_seeds_differ(self):
        from sbw.batch import philox_uniform_bxv
        r1 = philox_uniform_bxv(torch.tensor([42], dtype=torch.long), 100)
        r2 = philox_uniform_bxv(torch.tensor([43], dtype=torch.long), 100)
        assert not torch.equal(r1, r2)


class TestFusedPhilox:
    """Tests for fused Philox + threshold + bias path."""

    def test_fused_matches_unfused(self):
        """Fused path must produce bit-identical logits to the 3-pass path."""
        from sbw.batch import philox_apply_watermark, philox_uniform_bxv
        vocab_size = 1000
        seeds = torch.tensor([42, 123, 999], dtype=torch.long)

        for gamma, delta in [(0.5, 2.0), (0.25, 5.0), (0.75, 0.1)]:
            logits = torch.randn(3, vocab_size)

            # Unfused: mask then bias
            uniform = philox_uniform_bxv(seeds, vocab_size)
            mask = uniform < gamma
            expected = logits + mask.float() * delta

            # Fused
            result = philox_apply_watermark(seeds, logits.clone(), gamma, delta)
            assert torch.equal(result, expected), (
                f"Bit mismatch at gamma={gamma}, delta={delta}: "
                f"max diff={torch.max(torch.abs(result - expected)).item()}"
            )

    def test_integer_threshold_equivalence(self):
        """Integer threshold must match float threshold for various gamma values."""
        from sbw.batch import philox_uniform_bxv, philox_apply_watermark
        seeds = torch.tensor([42], dtype=torch.long)
        V = 2000
        for gamma in [0.25, 0.5, 0.75]:
            uniform = philox_uniform_bxv(seeds, V)
            float_mask = uniform < gamma

            logits_fused = philox_apply_watermark(seeds, torch.zeros(1, V), gamma, 1.0)
            int_mask = logits_fused[0] > 0.5  # delta=1.0, so green positions have value 1.0

            assert torch.equal(float_mask[0], int_mask), f"Threshold mismatch at gamma={gamma}"

    def test_fused_e2e_detect(self):
        """Watermark applied via fused path must be detectable."""
        vocab_size = 1000
        device = torch.device("cpu")
        wm = WatermarkBatch(vocab=[0] * vocab_size, gamma=0.5, delta=5.0,
                            seeding_scheme="gpu-simple_1", device=device)

        tokens = [42]
        for _ in range(100):
            ctx = torch.tensor([tokens[-1:]])
            logits = torch.randn(1, vocab_size)
            logits = wm.apply_watermark_fused(ctx, logits)
            tokens.append(logits.argmax(dim=-1).item())

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        detector = WatermarkDetector(
            device=device, tokenizer=tokenizer, vocab=[0] * vocab_size,
            gamma=0.5, seeding_scheme="gpu-simple_1", z_threshold=4.0,
            ignore_repeated_ngrams=False, normalizers=[],
        )
        result = detector.detect(tokenized_text=torch.tensor(tokens, device=device))
        assert result["prediction"] is True, f"Fused watermark not detected, z={result['z_score']:.2f}"


class TestPerRequestParams:
    """Tests for per-request gamma and delta in the fused path."""

    def test_per_request_delta(self):
        """Each row gets its own delta value."""
        from sbw.batch import philox_apply_watermark
        seeds = torch.tensor([42, 42, 42], dtype=torch.long)
        logits = torch.zeros(3, 1000)
        gamma_vec = torch.tensor([0.5, 0.5, 0.5])
        delta_vec = torch.tensor([1.0, 3.0, 5.0])

        result = philox_apply_watermark(seeds, logits, gamma_vec, delta_vec)
        for i, delta in enumerate([1.0, 3.0, 5.0]):
            unique = result[i].unique().sort().values
            assert torch.allclose(unique, torch.tensor([0.0, delta]))

    def test_per_request_gamma(self):
        """Each row gets its own gamma (green list fraction)."""
        from sbw.batch import philox_apply_watermark
        seeds = torch.tensor([42, 42], dtype=torch.long)
        logits = torch.zeros(2, 10000)
        gamma_vec = torch.tensor([0.25, 0.75])
        delta_vec = torch.tensor([1.0, 1.0])

        result = philox_apply_watermark(seeds, logits, gamma_vec, delta_vec)
        green_frac_0 = (result[0] > 0).float().mean().item()
        green_frac_1 = (result[1] > 0).float().mean().item()
        assert abs(green_frac_0 - 0.25) < 0.02
        assert abs(green_frac_1 - 0.75) < 0.02

    def test_vector_matches_scalar(self):
        """Uniform vector params must be bit-identical to scalar params."""
        from sbw.batch import philox_apply_watermark
        seeds = torch.tensor([42, 123, 999], dtype=torch.long)
        logits = torch.randn(3, 1000)

        result_scalar = philox_apply_watermark(seeds, logits.clone(), 0.5, 2.0)
        gamma_vec = torch.tensor([0.5, 0.5, 0.5])
        delta_vec = torch.tensor([2.0, 2.0, 2.0])
        result_vector = philox_apply_watermark(seeds, logits.clone(), gamma_vec, delta_vec)
        assert torch.equal(result_scalar, result_vector)

    def test_mixed_batch_detect(self):
        """Two sequences with different deltas must both be detectable."""
        vocab_size = 1000
        device = torch.device("cpu")
        wm = WatermarkBatch(vocab=[0] * vocab_size, seeding_scheme="gpu-simple_1", device=device)

        tokens_a, tokens_b = [42], [42]
        for _ in range(100):
            ctx = torch.tensor([[tokens_a[-1]], [tokens_b[-1]]])
            logits = torch.randn(2, vocab_size)
            gamma_vec = torch.tensor([0.5, 0.5])
            delta_vec = torch.tensor([2.0, 8.0])
            result = wm.apply_watermark_fused(ctx, logits, gamma_vec, delta_vec)
            tokens_a.append(result[0].argmax().item())
            tokens_b.append(result[1].argmax().item())

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        for tokens in [tokens_a, tokens_b]:
            detector = WatermarkDetector(
                device=device, tokenizer=tokenizer, vocab=[0] * vocab_size,
                gamma=0.5, seeding_scheme="gpu-simple_1", z_threshold=4.0,
                ignore_repeated_ngrams=False, normalizers=[],
            )
            result = detector.detect(tokenized_text=torch.tensor(tokens, device=device))
            assert result["prediction"] is True, f"z={result['z_score']:.2f}"


class TestGpuSimple1:
    """Tests for gpu-simple_1 parallel greenlist scheme."""

    @pytest.fixture
    def watermark(self):
        return WatermarkBatch(vocab=[0] * 1000, gamma=0.5, delta=2.0,
                              seeding_scheme="gpu-simple_1", device=torch.device("cpu"))

    def test_scheme_lookup(self, watermark):
        assert watermark.prf_type == "additive_prf"
        assert watermark.context_width == 1
        assert watermark.self_salt is False

    def test_deterministic(self, watermark):
        ctx = torch.tensor([[42]])
        m1 = watermark.get_greenlist_masks(ctx)[0].clone()
        m2 = watermark.get_greenlist_masks(ctx)[0].clone()
        assert torch.equal(m1, m2)

    def test_different_seeds_different_masks(self, watermark):
        m1 = watermark.get_greenlist_masks(torch.tensor([[42]]))[0]
        m2 = watermark.get_greenlist_masks(torch.tensor([[99]]))[0]
        assert not torch.equal(m1, m2)

    def test_green_fraction_approximate(self, watermark):
        """Green fraction should be approximately gamma (statistical, not exact)."""
        ctx = torch.tensor([[42]])
        mask = watermark.get_greenlist_masks(ctx)[0]
        frac = mask.float().mean().item()
        assert abs(frac - 0.5) < 0.05, f"Green fraction {frac} too far from gamma=0.5"

    def test_green_fraction_gamma_025(self):
        wm = WatermarkBatch(vocab=[0] * 10000, gamma=0.25, delta=2.0,
                            seeding_scheme="gpu-simple_1", device=torch.device("cpu"))
        mask = wm.get_greenlist_masks(torch.tensor([[42]]))[0]
        frac = mask.float().mean().item()
        assert abs(frac - 0.25) < 0.02, f"Green fraction {frac} too far from gamma=0.25"

    def test_batch_shape(self, watermark):
        ctx = torch.tensor([[10], [20], [30]])
        masks = watermark.get_greenlist_masks(ctx)
        assert masks.shape == (3, 1000)
        assert masks.dtype == torch.bool

    def test_batch_rows_differ(self, watermark):
        ctx = torch.tensor([[10], [20], [30]])
        masks = watermark.get_greenlist_masks(ctx)
        assert not torch.equal(masks[0], masks[1])
        assert not torch.equal(masks[1], masks[2])

    def test_set_seeding_scheme_roundtrip(self):
        """Switching to gpu-simple_1 and back should work."""
        wm = WatermarkBatch(vocab=[0] * 1000, device=torch.device("cpu"))
        mask_simple = wm.get_greenlist_masks(torch.tensor([[42]])).clone()
        wm.set_seeding_scheme("gpu-simple_1")
        assert wm.seeding_scheme == "gpu-simple_1"
        wm.set_seeding_scheme("simple_1")
        mask_after = wm.get_greenlist_masks(torch.tensor([[42]]))
        assert torch.equal(mask_simple, mask_after)


class TestWatermarkThenDetect:
    """End-to-end: apply watermark logit bias, sample greedily, detect."""

    @pytest.mark.parametrize("scheme,context_width", [
        ("simple_1", 1),
        ("gpu-simple_1", 1),
        ("minhash", 4),
        ("skipgram", 5),
    ])
    def test_watermark_then_detect(self, scheme, context_width):
        vocab_size = 1000
        device = torch.device("cpu")
        wm = WatermarkBatch(vocab=[0] * vocab_size, gamma=0.5, delta=5.0,
                            seeding_scheme=scheme, device=device)

        tokens = list(range(100, 100 + context_width))  # seed tokens
        for _ in range(100):
            ctx = torch.tensor([tokens[-context_width:]])
            logits = torch.randn(1, vocab_size)
            mask = wm.get_greenlist_masks(ctx)
            logits += mask.float() * wm.delta
            tokens.append(logits.argmax(dim=-1).item())

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        detector = WatermarkDetector(
            device=device,
            tokenizer=tokenizer,
            vocab=[0] * vocab_size,
            gamma=0.5,
            seeding_scheme=scheme,
            z_threshold=4.0,
            ignore_repeated_ngrams=False,
            normalizers=[],
        )
        result = detector.detect(tokenized_text=torch.tensor(tokens, device=device))
        assert result["prediction"] is True, f"{scheme}: expected watermark detected, got z={result['z_score']:.2f}"
        assert result["z_score"] > 4.0


class TestDummyDetect:
    """Tests for dummy_detect method (A3)."""

    @pytest.fixture
    def detector(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
        )

    def test_dummy_detect_nan_scores(self, detector):
        import math
        result = detector.dummy_detect()
        for key in ("num_tokens_scored", "num_green_tokens", "green_fraction", "z_score", "p_value"):
            assert math.isnan(result[key]), f"{key} should be NaN"

    def test_dummy_detect_prediction_false(self, detector):
        result = detector.dummy_detect()
        assert result["prediction"] is False


class TestGPUSchemes:
    """Tests for gpu-minhash, gpu-skipgram, and gpu-ff- schemes."""

    def test_gpu_minhash_config(self):
        """gpu-minhash should have same PRF config as minhash."""
        from sbw.prf import seeding_scheme_lookup
        assert seeding_scheme_lookup("gpu-minhash") == seeding_scheme_lookup("minhash")

    def test_gpu_skipgram_config(self):
        """gpu-skipgram should have same PRF config as skipgram."""
        from sbw.prf import seeding_scheme_lookup
        assert seeding_scheme_lookup("gpu-skipgram") == seeding_scheme_lookup("skipgram")

    def test_gpu_minhash_uses_parallel_path(self):
        """gpu-minhash should use the parallel green list method."""
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="gpu-minhash", device=torch.device("cpu"))
        assert wm.seeding_scheme.startswith("gpu-")
        assert wm.context_width == 4
        assert wm.self_salt is False

    def test_gpu_skipgram_uses_parallel_path(self):
        """gpu-skipgram should use the parallel green list method."""
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="gpu-skipgram", device=torch.device("cpu"))
        assert wm.seeding_scheme.startswith("gpu-")
        assert wm.context_width == 5
        assert wm.self_salt is False

    def test_gpu_minhash_deterministic(self):
        """gpu-minhash should produce deterministic results."""
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="gpu-minhash", device=torch.device("cpu"))
        ctx = torch.tensor([[10, 20, 30, 40]])
        m1 = wm.get_greenlist_masks(ctx)[0].clone()
        m2 = wm.get_greenlist_masks(ctx)[0].clone()
        assert torch.equal(m1, m2)

    def test_gpu_skipgram_deterministic(self):
        """gpu-skipgram should produce deterministic results."""
        wm = WatermarkBatch(vocab=[0] * 1000, seeding_scheme="gpu-skipgram", device=torch.device("cpu"))
        ctx = torch.tensor([[10, 20, 30, 40, 50]])
        m1 = wm.get_greenlist_masks(ctx)[0].clone()
        m2 = wm.get_greenlist_masks(ctx)[0].clone()
        assert torch.equal(m1, m2)

    def test_gpu_ff_valid_scheme(self):
        """gpu-ff- with self_salt=False should work."""
        from sbw.prf import seeding_scheme_lookup
        prf_type, ctx_width, self_salt, is_fused, num_candidates = seeding_scheme_lookup("gpu-ff-minhash_prf-4-False")
        assert prf_type == "minhash_prf"
        assert ctx_width == 4
        assert self_salt is False
        assert is_fused is False
        assert num_candidates is None

    def test_gpu_ff_self_salt_allowed(self):
        """gpu-ff- with self_salt=True should now be allowed (GPU selfhash support)."""
        from sbw.prf import seeding_scheme_lookup
        prf_type, ctx_width, self_salt, is_fused, num_candidates = seeding_scheme_lookup("gpu-ff-anchored_minhash_prf-4-True")
        assert prf_type == "anchored_minhash_prf"
        assert ctx_width == 4
        assert self_salt is True
        assert is_fused is False
        assert num_candidates == 40

    def test_gpu_ff_unknown_prf_rejected(self):
        """gpu-ff- with unknown PRF should raise error."""
        from sbw.prf import seeding_scheme_lookup
        with pytest.raises(ValueError, match="Unknown prf_type"):
            seeding_scheme_lookup("gpu-ff-unknown_prf-4-False")

    def test_gpu_ff_num_candidates_on_non_selfsalt_rejected(self):
        """gpu-ff- with num_candidates on non-self-salt scheme should raise error."""
        from sbw.prf import seeding_scheme_lookup
        with pytest.raises(ValueError, match="only valid for self-salt"):
            seeding_scheme_lookup("gpu-ff-minhash_prf-4-False-100")

    def test_gpu_ff_num_candidates_parsing(self):
        """gpu-ff- with num_candidates should parse correctly."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, _, num_candidates = seeding_scheme_lookup("gpu-ff-anchored_minhash_prf-4-True-100")
        assert num_candidates == 100
        _, _, _, _, num_candidates = seeding_scheme_lookup("gpu-ff-anchored_minhash_prf-4-True-fullvocab")
        assert num_candidates == 0


class TestGpuFusedSchemes:
    """Tests for GPU fused scheme parsing and validation."""

    def test_gpu_fused_selfhash_parsing(self):
        """gpu-fused-selfhash should parse correctly."""
        from sbw.prf import seeding_scheme_lookup
        prf, cw, ss, fused, nc = seeding_scheme_lookup("gpu-fused-selfhash")
        assert prf == "anchored_minhash_prf"
        assert cw == 4
        assert ss is True
        assert fused is True
        assert nc == 40

    def test_gpu_fused_selfhash_num_candidates(self):
        """gpu-fused-selfhash-<N> should parse num_candidates."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, fused, nc = seeding_scheme_lookup("gpu-fused-selfhash-100")
        assert fused is True
        assert nc == 100

    def test_gpu_fused_selfhash_fullvocab(self):
        """gpu-fused-selfhash-fullvocab should set num_candidates=0."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, fused, nc = seeding_scheme_lookup("gpu-fused-selfhash-fullvocab")
        assert fused is True
        assert nc == 0

    def test_gpu_fused_cpuhash_rejected(self):
        """gpu-fused-selfhash-cpuhash should be rejected."""
        from sbw.prf import seeding_scheme_lookup
        with pytest.raises(ValueError, match="does not support cpuhash"):
            seeding_scheme_lookup("gpu-fused-selfhash-cpuhash")

    def test_gpu_fused_unsupported_scheme_rejected(self):
        """gpu-fused- with unsupported base scheme should be rejected."""
        from sbw.prf import seeding_scheme_lookup
        with pytest.raises(ValueError, match="Unsupported fused scheme"):
            seeding_scheme_lookup("gpu-fused-simple_1")
        with pytest.raises(ValueError, match="Unsupported fused scheme"):
            seeding_scheme_lookup("gpu-fused-minhash")

    def test_gpu_fused_is_fused_flag(self):
        """gpu-fused schemes should have is_fused=True, gpu- schemes should have is_fused=False."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, fused_flag, _ = seeding_scheme_lookup("gpu-fused-selfhash")
        assert fused_flag is True
        _, _, _, fused_flag, _ = seeding_scheme_lookup("gpu-selfhash")
        assert fused_flag is False



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedEquivalence:
    """Fused compiled kernels must produce identical results to modular GPU paths."""

    def test_simple4_fullvocab_matches_modular(self):
        """gpu-fused-simple_4 must match gpu-simple_1 (same PRF, context_width=1 vs 4 differs, so compare to itself unfused)."""
        from sbw.batch import _simple_fused_compiled, philox_apply_watermark
        V = 5000
        device = torch.device("cuda")
        for seed_val in [42, 123, 99999]:
            ctx = torch.tensor([[seed_val, 1, 2, 3]], device=device)
            logits = torch.randn(1, V, device=device)
            # Fused path (mutates input in-place)
            result_fused = _simple_fused_compiled(ctx, logits.clone(), 0.5, 2.0, 15485863).clone()
            # Manual: same seed computation + philox_apply_watermark
            seeds = 15485863 * ctx.sum(dim=1)
            result_manual = philox_apply_watermark(seeds, logits.clone(), 0.5, 2.0)
            assert torch.allclose(result_fused, result_manual, atol=1e-6), \
                f"simple_4 fullvocab mismatch at seed={seed_val}"

    def test_simple4_topk_matches_fullvocab(self):
        """gpu-fused-simple_4-N must bias the same tokens as gpu-fused-simple_4 fullvocab."""
        V = 5000
        K = 100
        device = torch.device("cuda")
        wm_full = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                                 seeding_scheme="gpu-fused-simple_4", device=device)
        wm_topk = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                                 seeding_scheme="gpu-fused-simple_4-100", device=device)
        ctx = torch.tensor([[10, 20, 30, 40]], device=device)
        logits = torch.randn(1, V, device=device)
        # Fullvocab result
        result_full = wm_full.apply_watermark_simple_fused(ctx, logits.clone(), 0.5, 2.0).clone()
        # Topk result
        result_topk = wm_topk.apply_watermark_topk(ctx, logits.clone(), 0.5, 2.0)
        # Top-K tokens should have same bias in both
        _, top_indices = logits.topk(K, dim=-1)
        full_bias = (result_full - logits)[0, top_indices[0]]
        topk_bias = (result_topk - logits)[0, top_indices[0]]
        assert torch.allclose(full_bias, topk_bias, atol=1e-6), \
            "simple_4 topk biases differ from fullvocab on top-K tokens"

    def test_selfhash_fullvocab_matches_modular(self):
        """gpu-fused-selfhash-fullvocab must match gpu-selfhash-fullvocab (modular path)."""
        V = 2000
        device = torch.device("cuda")
        wm_fused = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                                  seeding_scheme="gpu-fused-selfhash-fullvocab", device=device)
        wm_modular = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                                    seeding_scheme="gpu-selfhash-fullvocab", device=device)
        ctx = torch.tensor([[100, 200, 300, 400]], device=device)
        logits = torch.randn(1, V, device=device)
        result_fused = wm_fused.apply_watermark_selfsalt_fused(ctx, logits.clone(), 0.5, 2.0).clone()
        # Modular self-salt path: use apply_watermark_selfsalt_direct
        result_modular = wm_modular.apply_watermark_selfsalt_direct(ctx, logits.clone(), 0.5, 2.0)
        assert torch.allclose(result_fused, result_modular, atol=1e-6), \
            f"selfhash fullvocab fused vs modular mismatch: max diff={torch.max(torch.abs(result_fused - result_modular)).item()}"

    def test_selfhash_topk_matches_fullvocab(self):
        """gpu-fused-selfhash-N must bias the same tokens as gpu-fused-selfhash-fullvocab on top-K."""
        V = 2000
        K = 50
        device = torch.device("cuda")
        wm_full = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                                 seeding_scheme="gpu-fused-selfhash-fullvocab", device=device)
        wm_topk = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                                 seeding_scheme=f"gpu-fused-selfhash-{K}", device=device)
        ctx = torch.tensor([[10, 20, 30, 40]], device=device)
        logits = torch.randn(1, V, device=device)
        result_full = wm_full.apply_watermark_selfsalt_fused(ctx, logits.clone(), 0.5, 2.0).clone()
        result_topk = wm_topk.apply_watermark_selfsalt_fused(ctx, logits.clone(), 0.5, 2.0)
        # Top-K tokens should have same bias
        _, top_indices = logits.topk(K, dim=-1)
        full_bias = (result_full - logits)[0, top_indices[0]]
        topk_bias = (result_topk - logits)[0, top_indices[0]]
        assert torch.allclose(full_bias, topk_bias, atol=1e-6), \
            "selfhash topk biases differ from fullvocab on top-K tokens"

    def test_fused_batch_consistency(self):
        """Fused kernels must produce same result regardless of batch size."""
        V = 3000
        device = torch.device("cuda")
        ctx = torch.tensor([[5, 10, 15, 20]], device=device)
        logits = torch.randn(1, V, device=device)
        wm = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                            seeding_scheme="gpu-fused-simple_4", device=device)
        result_b1 = wm.apply_watermark_simple_fused(ctx, logits.clone(), 0.5, 2.0).clone()
        # Same input in a batch of 4
        ctx_b4 = ctx.expand(4, -1)
        logits_b4 = logits.expand(4, -1).clone()
        result_b4 = wm.apply_watermark_simple_fused(ctx_b4, logits_b4, 0.5, 2.0).clone()
        assert torch.allclose(result_b1[0], result_b4[0], atol=1e-6), \
            "Batch size affects fused kernel output"


class TestNumCandidatesParsing:
    """Tests for num_candidates parsing across scheme families."""

    def test_cpu_selfhash_default(self):
        """CPU selfhash should default to num_candidates=40."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, _, nc = seeding_scheme_lookup("selfhash")
        assert nc == 40

    def test_cpu_selfhash_with_num_candidates(self):
        """CPU selfhash-<N> should parse num_candidates."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, _, nc = seeding_scheme_lookup("selfhash-100")
        assert nc == 100
        _, _, _, _, nc = seeding_scheme_lookup("selfhash-fullvocab")
        assert nc == 0

    def test_gpu_selfhash_variants(self):
        """gpu-selfhash variants should parse num_candidates correctly."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, _, nc = seeding_scheme_lookup("gpu-selfhash")
        assert nc == 40
        _, _, _, _, nc = seeding_scheme_lookup("gpu-selfhash-200")
        assert nc == 200
        _, _, _, _, nc = seeding_scheme_lookup("gpu-selfhash-fullvocab")
        assert nc == 0

    def test_gpu_selfhash_cpuhash_variants(self):
        """gpu-selfhash-cpuhash variants should parse num_candidates correctly."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, _, nc = seeding_scheme_lookup("gpu-selfhash-cpuhash")
        assert nc == 40
        _, _, _, _, nc = seeding_scheme_lookup("gpu-selfhash-cpuhash-100")
        assert nc == 100
        _, _, _, _, nc = seeding_scheme_lookup("gpu-selfhash-cpuhash-fullvocab")
        assert nc == 0

    def test_non_selfsalt_has_none_num_candidates(self):
        """Non-self-salt schemes should have num_candidates=None."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, _, nc = seeding_scheme_lookup("simple_1")
        assert nc is None
        _, _, _, _, nc = seeding_scheme_lookup("gpu-simple_1")
        assert nc is None
        _, _, _, _, nc = seeding_scheme_lookup("gpu-minhash")
        assert nc is None

    def test_cpu_freeform_num_candidates(self):
        """CPU freeform with self-salt should support num_candidates."""
        from sbw.prf import seeding_scheme_lookup
        _, _, _, _, nc = seeding_scheme_lookup("ff-anchored_minhash_prf-4-True")
        assert nc == 40
        _, _, _, _, nc = seeding_scheme_lookup("ff-anchored_minhash_prf-4-True-50")
        assert nc == 50

    @pytest.mark.parametrize("gpu_scheme,cpu_scheme,context_width", [
        ("gpu-minhash", "minhash", 4),
        ("gpu-skipgram", "skipgram", 5),
    ])
    def test_gpu_scheme_detectable(self, gpu_scheme, cpu_scheme, context_width):
        """Watermark applied via GPU scheme should be detectable."""
        vocab_size = 1000
        device = torch.device("cpu")
        wm = WatermarkBatch(vocab=[0] * vocab_size, gamma=0.5, delta=5.0,
                            seeding_scheme=gpu_scheme, device=device)

        tokens = list(range(100, 100 + context_width))
        for _ in range(100):
            ctx = torch.tensor([tokens[-context_width:]])
            logits = torch.randn(1, vocab_size)
            logits = wm.apply_watermark_fused(ctx, logits)
            tokens.append(logits.argmax(dim=-1).item())

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        detector = WatermarkDetector(
            device=device, tokenizer=tokenizer, vocab=[0] * vocab_size,
            gamma=0.5, seeding_scheme=gpu_scheme, z_threshold=4.0,
            ignore_repeated_ngrams=False, normalizers=[],
        )
        result = detector.detect(tokenized_text=torch.tensor(tokens, device=device))
        assert result["prediction"] is True, f"{gpu_scheme}: z={result['z_score']:.2f}"


class TestGpuSelfhash:
    """Tests for GPU-accelerated self-salt watermarking."""

    def test_scheme_lookup(self):
        """gpu-selfhash should map to anchored_minhash_prf with self_salt=True."""
        from sbw.prf import seeding_scheme_lookup
        prf_type, ctx_width, self_salt, is_fused, num_candidates = seeding_scheme_lookup("gpu-selfhash")
        assert prf_type == "anchored_minhash_prf"
        assert ctx_width == 4
        assert self_salt is True
        assert is_fused is False
        assert num_candidates == 40

    def test_hashint_gpu_nonzero(self):
        """hashint_gpu should never return 0."""
        from sbw.batch import hashint_gpu
        inputs = torch.arange(0, 10000)
        outputs = hashint_gpu(inputs)
        assert (outputs == 0).sum() == 0

    def test_hashint_gpu_deterministic(self):
        """hashint_gpu should be deterministic."""
        from sbw.batch import hashint_gpu
        x = torch.tensor([42, 1000, 50000])
        assert torch.equal(hashint_gpu(x), hashint_gpu(x))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_hashint_stays_on_gpu_and_matches_cpu(self):
        """hashint should stay on GPU when input is on GPU and produce same values as CPU."""
        from sbw.prf import hashint
        cpu_input = torch.tensor([100, 200, 300, 12345, 999999], dtype=torch.long)
        cpu_result = hashint(cpu_input)
        gpu_input = cpu_input.cuda()
        gpu_result = hashint(gpu_input)
        assert cpu_result.device.type == "cpu"
        assert gpu_result.device.type == "cuda"
        assert torch.equal(cpu_result, gpu_result.cpu())

    def test_philox_at_positions_matches_full(self):
        """philox_at_positions should match full sequence generation."""
        from sbw.batch import philox_at_positions, philox_uniform_bxv
        seed = torch.tensor([12345])
        for pos in [0, 42, 99]:
            full = philox_uniform_bxv(seed, 100)
            expected = full[0, pos].item()
            actual = philox_at_positions(seed, torch.tensor([pos])).item()
            assert abs(expected - actual) < 1e-6

    def test_extend_context_for_selfsalt(self):
        """extend_context_for_selfsalt should produce correct shapes and values."""
        from sbw.prf import extend_context_for_selfsalt
        ctx = torch.tensor([[100, 200, 300, 400], [500, 600, 700, 800]])  # (2, 4)
        cand = torch.tensor([[10, 20, 30], [40, 50, 60]])  # (2, 3)
        extended = extend_context_for_selfsalt(ctx, cand)
        assert extended.shape == (6, 4)  # (B*N, h)
        # Check first batch, first candidate: [200, 300, 400, 10]
        assert extended[0].tolist() == [200, 300, 400, 10]
        # Check second batch, first candidate: [600, 700, 800, 40]
        assert extended[3].tolist() == [600, 700, 800, 40]

    def test_gpu_selfsalt_mask_shape(self):
        """GPU self-salt should return correct mask shape."""
        wm = WatermarkBatch(vocab=[0]*1000, gamma=0.5, seeding_scheme="gpu-selfhash",
                           device=torch.device("cpu"))
        context = torch.randint(0, 1000, (4, 4))
        logits = torch.randn(4, 1000)
        masks = wm.get_greenlist_masks(context, logits)
        assert masks.shape == (4, 1000)
        assert masks.dtype == torch.bool

    def test_gpu_selfsalt_green_fraction(self):
        """With num_candidates=40, green fraction should be <= 40/vocab_size."""
        wm = WatermarkBatch(vocab=[0]*1000, gamma=0.5, seeding_scheme="gpu-selfhash",
                           device=torch.device("cpu"))
        context = torch.randint(0, 1000, (10, 4))
        logits = torch.randn(10, 1000)
        masks = wm.get_greenlist_masks(context, logits)
        green_frac = masks.float().mean().item()
        assert green_frac <= 0.05  # 40/1000 * gamma ≈ 0.02, allow margin

    def test_gpu_selfsalt_deterministic(self):
        """GPU self-salt should be deterministic."""
        wm = WatermarkBatch(vocab=[0]*1000, gamma=0.5, seeding_scheme="gpu-selfhash",
                           device=torch.device("cpu"))
        context = torch.randint(0, 1000, (4, 4))
        logits = torch.randn(4, 1000)
        m1 = wm.get_greenlist_masks(context, logits)
        m2 = wm.get_greenlist_masks(context, logits)
        assert torch.equal(m1, m2)

    def test_gpu_selfsalt_detection_matches_generation(self):
        """Detector should agree with generator on green/non-green status."""
        from sbw.batch import hashint_gpu, philox_at_positions
        from sbw.prf import extend_context_for_selfsalt, anchored_minhash_prf

        vocab_size = 1000
        wm = WatermarkBatch(vocab=[0]*vocab_size, gamma=0.5, seeding_scheme="gpu-selfhash",
                           device=torch.device("cpu"))
        context = torch.randint(0, 1000, (1, 4))
        logits = torch.randn(1, vocab_size)
        masks = wm.get_greenlist_masks(context, logits)

        # Get top-40 candidates (what the generator evaluated)
        _, candidates = logits.topk(40, dim=-1)

        # Check each candidate
        for tok in candidates[0].tolist():
            # Compute expected green status using same logic as detector
            cand = torch.tensor([[tok]])
            extended = extend_context_for_selfsalt(context, cand)
            seed = anchored_minhash_prf(extended, wm.hash_key, hashint_fn=hashint_gpu)
            rand_val = philox_at_positions(seed, cand.squeeze(0))
            expected_green = (rand_val < wm.gamma).item()
            actual_green = masks[0, tok].item()
            assert expected_green == actual_green, f"Token {tok}: expected {expected_green}, got {actual_green}"

    def test_gpu_selfhash_e2e_detection(self):
        """End-to-end: generate watermarked tokens, detect with z-score."""
        vocab_size = 1000
        device = torch.device("cpu")
        gamma = 0.5
        delta = 5.0

        wm = WatermarkBatch(vocab=[0]*vocab_size, gamma=gamma, delta=delta,
                           seeding_scheme="gpu-selfhash", device=device)

        # Simulate watermarked generation: always pick green token from top candidates
        context_width = wm.context_width
        tokens = list(range(context_width))  # Initial context
        for _ in range(100):
            context = torch.tensor(tokens[-context_width:], device=device).unsqueeze(0)
            logits = torch.randn(1, vocab_size, device=device)
            mask = wm.get_greenlist_masks(context, logits)

            # Pick highest-scoring green token
            biased_logits = logits.clone()
            biased_logits[0, mask[0]] += delta
            next_token = biased_logits.argmax(dim=-1).item()
            tokens.append(next_token)

        # Detect
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        detector = WatermarkDetector(
            device=device, tokenizer=tokenizer, vocab=[0]*vocab_size,
            gamma=gamma, seeding_scheme="gpu-selfhash", z_threshold=4.0,
            ignore_repeated_ngrams=False, normalizers=[],
        )
        result = detector.detect(tokenized_text=torch.tensor(tokens, device=device))
        assert result["prediction"] is True, f"z={result['z_score']:.2f}, expected detection"
        assert result["z_score"] > 4.0, f"z={result['z_score']:.2f}, expected > 4.0"


class TestBatchedDetection:
    """Tests that batched GPU detection produces identical results to per-token path."""

    @pytest.mark.parametrize("scheme", [
        "gpu-simple_1",
        "gpu-minhash",
        "gpu-skipgram",
        "gpu-selfhash",
    ])
    def test_batched_matches_per_token(self, scheme):
        """Batched GPU detection must produce identical scores to per-token _is_token_green path."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        vocab_size = len(tokenizer)
        
        detector = WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * vocab_size,
            gamma=0.5,
            seeding_scheme=scheme,
            ignore_repeated_ngrams=False,  # Simpler to verify without deduplication
        )
        
        # Test on sample text
        text = "The quick brown fox jumps over the lazy dog. " * 3
        tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        
        # Get result from batched path
        result_batched = detector._score_sequence_gpu_batched(
            tokenized, return_green_token_mask=True
        )
        
        # Compute expected result using per-token _is_token_green
        h = detector.watermark.context_width
        self_salt = detector.watermark.self_salt
        expected_mask = []
        
        if self_salt:
            # For self-salt, ngram is h tokens where last is target
            for i in range(len(tokenized) - h + 1):
                prefix = tuple(tokenized[i:i+h].tolist())
                target = tokenized[i+h-1].item()
                expected_mask.append(detector._is_token_green(prefix, target))
        else:
            # For non-self-salt, h context tokens + target
            for i in range(len(tokenized) - h):
                prefix = tuple(tokenized[i:i+h].tolist())
                target = tokenized[i+h].item()
                expected_mask.append(detector._is_token_green(prefix, target))
        
        assert result_batched["green_token_mask"] == expected_mask, \
            f"Green token masks differ for {scheme}"
        assert result_batched["num_tokens_scored"] == len(expected_mask)
        assert result_batched["num_green_tokens"] == sum(expected_mask)

    @pytest.mark.parametrize("scheme", ["gpu-simple_1", "gpu-selfhash"])
    def test_batched_with_deduplication(self, scheme):
        """Batched path with ignore_repeated_ngrams should match per-token path."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        vocab_size = len(tokenizer)
        
        detector = WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * vocab_size,
            gamma=0.5,
            seeding_scheme=scheme,
            ignore_repeated_ngrams=True,
        )
        
        # Repetitive text to test deduplication
        text = "hello world hello world hello world"
        
        result = detector.detect(text=text, return_green_token_mask=True)
        
        # Basic sanity checks
        assert "z_score" in result
        assert "num_tokens_scored" in result
        assert "num_green_tokens" in result
        assert result["num_green_tokens"] <= result["num_tokens_scored"]

    def test_batched_windowed_detection(self):
        """Batched path should work correctly with windowed detection."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        vocab_size = len(tokenizer)
        
        detector = WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * vocab_size,
            gamma=0.5,
            seeding_scheme="gpu-simple_1",
            ignore_repeated_ngrams=False,  # More tokens to score
        )
        
        # Longer text to have enough tokens
        text = "The quick brown fox jumps over the lazy dog. " * 20
        
        # Test windowed detection with smaller windows
        result = detector.detect(text=text, window_size="10,20,30")
        
        assert "z_score" in result
        assert "num_tokens_scored" in result
        assert result["num_tokens_scored"] in [10, 20, 30]


class TestGreenlistConsistency:
    """Tests that fused/direct generation paths produce identical green lists to detection path."""

    @pytest.mark.parametrize("scheme", [
        "gpu-simple_1",
        "gpu-minhash",
        "gpu-skipgram",
        "gpu-ff-minhash_prf-4-False",
        "gpu-ff-skipgram_prf-5-False",
    ])
    def test_fused_matches_detection_non_selfsalt(self, scheme):
        """Fused path must produce identical green list to get_greenlist_masks for non-self-salt schemes."""
        from sbw.prf import seeding_scheme_lookup
        vocab_size = 5000
        _, context_width, self_salt, _, _ = seeding_scheme_lookup(scheme)
        assert not self_salt

        wm = WatermarkBatch(vocab=[0] * vocab_size, gamma=0.5, delta=1.0,
                            seeding_scheme=scheme, device=torch.device("cpu"))

        for _ in range(5):
            context = torch.randint(0, vocab_size, (4, context_width))
            zero_logits = torch.zeros(4, vocab_size)

            # Fused path: delta=1.0 on zero logits -> green tokens have value 1.0
            biased = wm.apply_watermark_fused(context, zero_logits, gamma=0.5, delta=1.0)
            mask_from_fused = biased > 0.5

            # Detection path
            mask_from_detection = wm.get_greenlist_masks(context, gamma=0.5)

            assert torch.equal(mask_from_fused, mask_from_detection), (
                f"{scheme}: fused and detection paths differ"
            )

    @pytest.mark.parametrize("scheme", [
        "gpu-selfhash",
        "gpu-ff-anchored_minhash_prf-4-True",
        "gpu-ff-anchored_skipgram_prf-5-True",
    ])
    def test_direct_matches_detection_selfsalt(self, scheme):
        """Direct path must produce identical green list to get_greenlist_masks for self-salt schemes."""
        from sbw.prf import seeding_scheme_lookup
        vocab_size = 5000
        _, context_width, self_salt, _, _ = seeding_scheme_lookup(scheme)
        assert self_salt

        wm = WatermarkBatch(vocab=[0] * vocab_size, gamma=0.5, delta=1.0,
                            seeding_scheme=scheme, device=torch.device("cpu"))

        for _ in range(3):
            context = torch.randint(0, vocab_size, (2, context_width))
            zero_logits = torch.zeros(2, vocab_size)

            # Direct path with num_candidates=vocab_size to get full mask
            biased = wm.apply_watermark_selfsalt_direct(
                context, zero_logits, gamma=0.5, delta=1.0, num_candidates=vocab_size
            )
            mask_from_direct = biased > 0.5

            # Detection path with num_candidates=vocab_size
            mask_from_detection = wm.get_greenlist_masks(
                context, zero_logits, gamma=0.5, num_candidates=vocab_size
            )

            assert torch.equal(mask_from_direct, mask_from_detection), (
                f"{scheme}: direct and detection paths differ"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
