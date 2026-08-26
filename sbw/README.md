# sbw

**Stateless Bernoulli Watermarking** — A fast, efficient watermark detection library for LLM-generated text.

This package provides watermark detection capabilities. For watermark injection during text generation, see [vllm-sbw](../vllm_sbw/).

## Features

- **GPU-accelerated detection** using PyTorch and custom Philox PRNG
- **Batch processing** for efficient detection on multiple texts
- **Multiple seeding schemes**: selfhash, lefthash, minhash, and additive
- **Text normalization** with Unicode and homoglyph handling

## Installation

```bash
pip install sbw
```

## Quick Start

```python
from transformers import AutoTokenizer
from sbw import WatermarkDetector
import torch

# Initialize detector
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
detector = WatermarkDetector(
    device=torch.device("cuda"),
    tokenizer=tokenizer,
    vocab=list(range(len(tokenizer))),
    gamma=0.25,
    seeding_scheme="selfhash",
    hash_key=15485863,
    z_threshold=4.0,
)

# Detect watermark
text = "Your text to analyze..."
result = detector.detect(text=text)

print(f"Watermark detected: {result['prediction']}")
print(f"Z-score: {result['z_score']:.2f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Green fraction: {result['green_fraction']:.2%}")
```

## Batch Detection

For processing multiple texts efficiently:

```python
from sbw import WatermarkBatch

# Create batch processor
batch = WatermarkBatch(
    vocab_size=len(tokenizer),
    gamma=0.25,
    seeding_scheme="selfhash",
    hash_key=15485863,
    device=torch.device("cuda"),
)

# Process multiple texts
texts = ["Text 1...", "Text 2...", "Text 3..."]
token_ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

results = batch.detect_batch(token_ids)
for i, result in enumerate(results):
    print(f"Text {i+1}: z={result['z_score']:.2f}, detected={result['prediction']}")
```

## Parameters

### Detection Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gamma` | Fraction of vocabulary in the "green" list | 0.25 |
| `seeding_scheme` | How to derive PRF seed from context | "selfhash" |
| `hash_key` | Secret key for watermark | 15485863 |
| `z_threshold` | Z-score threshold for detection | 4.0 |

### Seeding Schemes

- **selfhash**: Uses hash of current token (recommended for robustness)
- **lefthash**: Uses hash of previous token
- **minhash**: Minimum hash over context window
- **additive**: Sum of hashes over context window

## API Reference

### WatermarkDetector

Main detection class with text normalization and full detection pipeline.

```python
detector = WatermarkDetector(
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    vocab: List[int],
    gamma: float = 0.25,
    seeding_scheme: str = "selfhash",
    hash_key: int = 15485863,
    z_threshold: float = 4.0,
    normalizers: List[str] = ["unicode", "homoglyphs", "truecase"],
)

result = detector.detect(text: str) -> Dict[str, Any]
```

### WatermarkBatch

Low-level batch processor for efficient detection without tokenization overhead.

```python
batch = WatermarkBatch(
    vocab_size: int,
    gamma: float = 0.25,
    seeding_scheme: str = "selfhash",
    hash_key: int = 15485863,
    device: torch.device = None,
)

results = batch.detect_batch(token_ids: List[List[int]]) -> List[Dict]
```

## Related Projects

- [vllm-sbw](../vllm_sbw/) — vLLM integration for watermark injection
- [waterpipe](https://github.com/si-mon-jinn/waterpipe) — Evaluation pipeline
- [flip-dont-shuffle](https://github.com/si-mon-jinn/flip-dont-shuffle) — Paper repository
- [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) — KGW watermarking (Kirchenbauer et al.)

## Citation

```bibtex
@inproceedings{flip-dont-shuffle-2026,
  title={Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference},
  author={Ceppi, Simone and Sanchez, Ignacio},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

## License

EUPL-1.2
