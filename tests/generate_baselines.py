#!/usr/bin/env python3
"""Generate baseline detection values. Run once before refactoring."""

import json
import torch
from transformers import AutoTokenizer
from watermark import WatermarkDetector

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
detector = WatermarkDetector(
    device=torch.device("cpu"),
    tokenizer=tokenizer,
    vocab=[0] * len(tokenizer),
    gamma=0.5,
    z_threshold=4.0,
    ignore_repeated_bigrams=False,
)

test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can generate human-like text.",
    "Python is a popular programming language for data science.",
    "Artificial intelligence is transforming many industries.",
    "The watermark detection algorithm uses statistical analysis.",
]

baselines = {}
for text in test_texts:
    result = detector.detect(text=text)
    baselines[text] = {
        "z_score": result["z_score"],
        "green_fraction": result["green_fraction"],
        "num_tokens_scored": result["num_tokens_scored"],
        "num_green_tokens": result["num_green_tokens"],
    }

import os
baseline_path = os.path.join(os.path.dirname(__file__), "baselines.json")
with open(baseline_path, "w") as f:
    json.dump(baselines, f, indent=2)

print(f"Generated baselines for {len(baselines)} texts")
