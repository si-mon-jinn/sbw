"""
Cross-library compatibility tests: sbw vs lm-watermarking.

Verifies that CPU seeding schemes (simple_1, selfhash, minhash, skipgram)
produce identical results between the two implementations.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Load lm-watermarking modules dynamically
LM_WATERMARKING_PATH = Path(__file__).resolve().parent.parent.parent / "lm-watermarking"

if not LM_WATERMARKING_PATH.exists():
    pytest.skip(
        f"lm-watermarking not found at {LM_WATERMARKING_PATH}. "
        "Clone https://github.com/jwkirchenbauer/lm-watermarking to run compatibility tests.",
        allow_module_level=True,
    )


def _load_lm_module(name: str):
    """Load a module from lm-watermarking directory."""
    # Add lm-watermarking to sys.path for internal imports
    lm_path_str = str(LM_WATERMARKING_PATH)
    if lm_path_str not in sys.path:
        sys.path.insert(0, lm_path_str)
    
    spec = importlib.util.spec_from_file_location(
        f"lm_{name}", LM_WATERMARKING_PATH / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"lm_{name}"] = module
    spec.loader.exec_module(module)
    return module


# Pre-load modules
lm_alternative_prf = _load_lm_module("alternative_prf_schemes")
lm_watermark_processor = _load_lm_module("watermark_processor")
lm_extended_watermark_processor = _load_lm_module("extended_watermark_processor")


class TestHashTableCompatibility:
    """Verify hash tables are identical between implementations."""

    def test_hash_table_identical(self):
        """Both libraries must use the same permutation table."""
        from sbw.prf import _fixed_table as sbw_table

        lm_table = lm_alternative_prf.fixed_table
        assert torch.equal(sbw_table, lm_table), "Hash tables differ"

    def test_hashint_identical(self):
        """hashint() must produce identical results."""
        from sbw.prf import hashint as sbw_hashint

        lm_hashint = lm_alternative_prf.hashint
        test_values = torch.tensor([0, 1, 42, 1000, 15485863, 2**31 - 1])
        
        sbw_result = sbw_hashint(test_values)
        lm_result = lm_hashint(test_values)

        assert torch.equal(sbw_result, lm_result), f"hashint differs: {sbw_result} vs {lm_result}"


class TestPRFCompatibility:
    """Verify PRF functions produce identical seeds."""

    @pytest.fixture
    def test_contexts(self):
        """Sample contexts for testing."""
        return [
            torch.tensor([[100]]),  # simple_1: context_width=1
            torch.tensor([[10, 20, 30, 40]]),  # selfhash/minhash: context_width=4
            torch.tensor([[10, 20, 30, 40, 50]]),  # skipgram: context_width=5
        ]

    @pytest.mark.parametrize("prf_name", [
        "additive_prf",
        "multiplicative_prf",
        "minfunc_prf",
        "minhash_prf",
        "skipgram_prf",
        "anchored_minhash_prf",
    ])
    def test_prf_identical(self, prf_name):
        """Each PRF must produce identical results."""
        from sbw.prf import prf_lookup as sbw_prf_lookup

        lm_prf_lookup = lm_alternative_prf.prf_lookup
        sbw_prf = sbw_prf_lookup[prf_name]
        lm_prf = lm_prf_lookup[prf_name]

        # Use appropriate context width
        if prf_name in ("additive_prf", "multiplicative_prf", "minfunc_prf"):
            context = torch.tensor([[100, 200]])
        elif prf_name == "skipgram_prf":
            context = torch.tensor([[10, 20, 30, 40, 50]])
        else:
            context = torch.tensor([[10, 20, 30, 40]])

        salt_key = 15485863

        # sbw returns tensor, lm returns scalar
        sbw_result = sbw_prf(context, salt_key)
        lm_result = lm_prf(context[0], salt_key)

        assert sbw_result[0].item() == lm_result, \
            f"{prf_name}: sbw={sbw_result[0].item()}, lm={lm_result}"


class TestSeedingSchemeCompatibility:
    """Verify seeding_scheme_lookup returns compatible configurations."""

    @pytest.mark.parametrize("scheme,expected_prf,expected_width,expected_selfsalt", [
        ("simple_1", "additive_prf", 1, False),
        ("selfhash", "anchored_minhash_prf", 4, True),
        ("minhash", "minhash_prf", 4, False),
        ("skipgram", "skipgram_prf", 5, False),
    ])
    def test_scheme_lookup_compatible(self, scheme, expected_prf, expected_width, expected_selfsalt):
        """Scheme lookup must return compatible configurations."""
        from sbw.prf import seeding_scheme_lookup as sbw_lookup

        lm_lookup = lm_alternative_prf.seeding_scheme_lookup
        sbw_prf, sbw_width, sbw_selfsalt, _, _ = sbw_lookup(scheme)
        lm_prf, lm_width, lm_selfsalt, lm_hash_key = lm_lookup(scheme)

        assert sbw_prf == lm_prf == expected_prf
        assert sbw_width == lm_width == expected_width
        assert sbw_selfsalt == lm_selfsalt == expected_selfsalt


class TestGreenListCompatibility:
    """Verify green list generation produces identical results."""

    @pytest.fixture
    def vocab_size(self):
        return 1000

    @pytest.fixture
    def gamma(self):
        return 0.5

    def _get_lm_greenlist(self, context_token: int, vocab_size: int, gamma: float, 
                          hash_key: int = 15485863, device=torch.device("cpu")):
        """Get green list using lm-watermarking implementation."""
        WatermarkBase = lm_watermark_processor.WatermarkBase

        class LMWatermark(WatermarkBase):
            pass

        wm = LMWatermark(
            vocab=[0] * vocab_size,
            gamma=gamma,
            hash_key=hash_key,
        )
        wm.rng = torch.Generator(device=device)

        input_ids = torch.tensor([context_token], device=device)
        greenlist_ids = wm._get_greenlist_ids(input_ids)
        return set(greenlist_ids.tolist())

    def _get_sbw_greenlist(self, context_token: int, vocab_size: int, gamma: float,
                            hash_key: int = 15485863, device=torch.device("cpu")):
        """Get green list using sbw implementation."""
        from sbw import WatermarkBatch

        wm = WatermarkBatch(
            vocab=[0] * vocab_size,
            gamma=gamma,
            hash_key=hash_key,
            seeding_scheme="simple_1",
            device=device,
        )

        context = torch.tensor([[context_token]], device=device)
        mask = wm._get_greenlist_masks_standard(context)
        return set(torch.where(mask[0])[0].tolist())

    @pytest.mark.parametrize("context_token", [0, 1, 42, 100, 999, 15485863])
    def test_greenlist_identical_simple_1(self, context_token, vocab_size, gamma):
        """Green lists must be identical for simple_1 scheme."""
        lm_greenlist = self._get_lm_greenlist(context_token, vocab_size, gamma)
        sbw_greenlist = self._get_sbw_greenlist(context_token, vocab_size, gamma)

        assert lm_greenlist == sbw_greenlist, \
            f"Green lists differ for context_token={context_token}"

    def test_greenlist_size_correct(self, vocab_size, gamma):
        """Green list size must be gamma * vocab_size."""
        expected_size = int(vocab_size * gamma)

        lm_greenlist = self._get_lm_greenlist(42, vocab_size, gamma)
        sbw_greenlist = self._get_sbw_greenlist(42, vocab_size, gamma)

        assert len(lm_greenlist) == expected_size
        assert len(sbw_greenlist) == expected_size

    def _is_green_selfhash_lm(self, context: torch.Tensor, candidate: int, 
                              vocab_size: int, gamma: float):
        """Check if candidate is green using lm-watermarking selfhash."""
        WatermarkBase = lm_extended_watermark_processor.WatermarkBase
        
        class LMWatermark(WatermarkBase):
            pass
        
        wm = LMWatermark(
            vocab=[0] * vocab_size,
            gamma=gamma,
            seeding_scheme="selfhash",
        )
        wm.rng = torch.Generator(device=torch.device("cpu"))
        
        # Selfhash: extend context with candidate, then check if candidate in greenlist
        extended = torch.cat([context, torch.tensor([candidate])])
        greenlist_ids = wm._get_greenlist_ids(extended)
        return candidate in greenlist_ids.tolist()

    def _is_green_selfhash_sbw(self, context: torch.Tensor, candidate: int,
                                vocab_size: int, gamma: float):
        """Check if candidate is green using sbw selfhash."""
        from sbw import WatermarkBatch
        
        wm = WatermarkBatch(
            vocab=[0] * vocab_size,
            gamma=gamma,
            seeding_scheme="selfhash",
            device=torch.device("cpu"),
        )
        # Create fake logits with candidate as top token to trigger selfhash evaluation
        logits = torch.zeros(1, vocab_size)
        logits[0, candidate] = 100.0  # Make candidate the top token
        
        # Need at least context_width tokens; pad if needed
        if len(context) < wm.context_width:
            context = torch.cat([torch.zeros(wm.context_width - len(context), dtype=torch.long), context])
        context = context[-wm.context_width:].unsqueeze(0)
        
        mask = wm._get_greenlist_masks_selfhash(context, logits)
        return mask[0, candidate].item()

    @pytest.mark.parametrize("context,candidate", [
        (torch.tensor([100, 200, 300]), 42),
        (torch.tensor([100, 200, 300]), 500),
        (torch.tensor([10, 20, 30]), 999),
        (torch.tensor([1, 2, 3]), 0),
        (torch.tensor([500, 600, 700]), 123),
    ])
    def test_greenlist_identical_selfhash(self, context, candidate, vocab_size, gamma):
        """Selfhash green list membership must be identical."""
        lm_is_green = self._is_green_selfhash_lm(context, candidate, vocab_size, gamma)
        sbw_is_green = self._is_green_selfhash_sbw(context, candidate, vocab_size, gamma)
        
        assert lm_is_green == sbw_is_green, \
            f"Selfhash differs for context={context.tolist()}, candidate={candidate}: lm={lm_is_green}, sbw={sbw_is_green}"


class TestDetectionCompatibility:
    """Verify detection produces identical z-scores."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("gpt2")

    @pytest.fixture
    def vocab_size(self, tokenizer):
        return len(tokenizer)

    @pytest.fixture
    def test_texts(self):
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming how we build software.",
            "Python is a popular programming language for data science.",
        ]

    def test_detection_z_scores_identical(self, tokenizer, vocab_size, test_texts):
        """Detection z-scores must be identical between implementations."""
        from sbw import WatermarkDetector as SBWDetector

        LMDetector = lm_watermark_processor.WatermarkDetector
        device = torch.device("cpu")
        gamma = 0.5
        hash_key = 15485863

        sbw_detector = SBWDetector(
            device=device,
            tokenizer=tokenizer,
            vocab=[0] * vocab_size,
            gamma=gamma,
            z_threshold=4.0,
            seeding_scheme="simple_1",
            hash_key=hash_key,
            ignore_repeated_bigrams=False,
            normalizers=[],
        )

        lm_detector = LMDetector(
            device=device,
            tokenizer=tokenizer,
            vocab=[0] * vocab_size,
            gamma=gamma,
            z_threshold=4.0,
            seeding_scheme="simple_1",
            hash_key=hash_key,
            ignore_repeated_bigrams=False,
            normalizers=[],
        )

        for text in test_texts:
            sbw_result = sbw_detector.detect(text=text)
            lm_result = lm_detector.detect(text=text)

            assert abs(sbw_result["z_score"] - lm_result["z_score"]) < 1e-6, \
                f"z_score differs for '{text[:30]}...': sbw={sbw_result['z_score']}, lm={lm_result['z_score']}"
            assert sbw_result["num_green_tokens"] == lm_result["num_green_tokens"], \
                f"num_green_tokens differs for '{text[:30]}...'"
            assert sbw_result["num_tokens_scored"] == lm_result["num_tokens_scored"], \
                f"num_tokens_scored differs for '{text[:30]}...'"
