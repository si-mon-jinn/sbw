"""
Compatibility tests to ensure detection results are preserved after refactoring.
"""

import torch
import pytest


class TestDetectorCompatibility:
    """Verify detector produces consistent results."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    @pytest.fixture
    def detector(self, tokenizer):
        from sbw import WatermarkDetector
        return WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
            z_threshold=4.0,
            ignore_repeated_bigrams=False,
        )

    def test_detection_returns_expected_keys(self, detector):
        """Detection result should contain expected fields."""
        result = detector.detect(text="This is a sample text for testing.")
        assert "z_score" in result
        assert "prediction" in result
        assert "num_tokens_scored" in result
        assert "green_fraction" in result
        assert "num_green_tokens" in result


class TestBaselinePreservation:
    """Verify detection results match stored baselines exactly."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    @pytest.fixture
    def detector(self, tokenizer):
        from sbw import WatermarkDetector
        return WatermarkDetector(
            device=torch.device("cpu"),
            tokenizer=tokenizer,
            vocab=[0] * len(tokenizer),
            gamma=0.5,
            z_threshold=4.0,
            ignore_repeated_bigrams=False,
        )

    def test_baselines_match(self, detector):
        """Detection results must match stored baselines exactly."""
        import json
        import os
        
        baseline_path = os.path.join(os.path.dirname(__file__), "baselines.json")
        if not os.path.exists(baseline_path):
            pytest.skip("baselines.json not found - run generate_baselines.py first")
        
        with open(baseline_path) as f:
            baselines = json.load(f)

        for text, expected in baselines.items():
            result = detector.detect(text=text)
            assert abs(result["z_score"] - expected["z_score"]) < 1e-6, \
                f"z_score mismatch for: {text[:30]}..."
            assert result["num_green_tokens"] == expected["num_green_tokens"], \
                f"num_green_tokens mismatch for: {text[:30]}..."
            assert result["num_tokens_scored"] == expected["num_tokens_scored"], \
                f"num_tokens_scored mismatch for: {text[:30]}..."
