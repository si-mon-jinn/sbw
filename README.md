# SBW — Stateless Bernoulli Watermarking

Fast, efficient watermarking for LLM-generated text. This repository contains two packages:

| Package | PyPI | Description |
|---------|------|-------------|
| [sbw](sbw/) | `pip install sbw` | Watermark detection library |
| [vllm-sbw](vllm_sbw/) | `pip install vllm-sbw` | vLLM integration for watermark injection |

## Overview

SBW implements the watermarking scheme from "Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference" (EMNLP 2026). It achieves near-zero inference overhead through:

- **Stateless design**: No state tracking across tokens
- **GPU-native computation**: Parallel PRF evaluation on GPU
- **CUDAGraph compatibility**: Works with vLLM optimizations

## Quick Start

### Detection Only

```bash
pip install sbw
```

```python
from sbw import WatermarkDetector
from transformers import AutoTokenizer
import torch

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

result = detector.detect(text="Your text here...")
print(f"Watermark: {result['prediction']}, z={result['z_score']:.2f}")
```

### Watermarked Generation

```bash
pip install vllm-sbw
```

```bash
# Start vLLM with watermarking
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --logits-processors vllm_sbw:SBWLogitsProcessor
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Write a story about:",
    max_tokens=200,
    extra_body={"vllm_xargs": {"gamma": 0.25, "delta": 2.0}}
)
```

## Repository Structure

```
sbw/
├── sbw/                    # Detection package
│   ├── pyproject.toml      # pip install sbw
│   └── sbw/                # Python module
│       ├── detector.py     # WatermarkDetector class
│       ├── batch.py        # Batch processing
│       ├── prf.py          # Philox PRNG implementation
│       └── normalizers/    # Text normalization
├── vllm_sbw/               # vLLM integration package
│   ├── pyproject.toml      # pip install vllm-sbw
│   └── vllm_sbw/           # Python module
│       └── logits_processor.py
├── examples/               # Usage examples
├── benchmarks/             # Performance benchmarks
└── tests/                  # Test suite
```

## Running Tests

```bash
pip install -e sbw/[dev]
pytest tests/
```

## Benchmarks

See the [benchmarks/](benchmarks/) directory for performance measurements and comparison scripts.

```bash
cd benchmarks
python profile_logits_processor.py
```

## Related Projects

- [waterpipe](https://github.com/si-mon-jinn/waterpipe) — Watermark evaluation pipeline
- [flip-dont-shuffle](https://github.com/si-mon-jinn/flip-dont-shuffle) — Paper repository with experiments

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

EUPL-1.2 — See [LICENSE](LICENSE.txt) for details.
