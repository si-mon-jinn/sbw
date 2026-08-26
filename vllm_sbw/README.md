# vllm-sbw

**SBW Watermark Injection for vLLM** — A logits processor that adds Stateless Bernoulli Watermarks during text generation.

This package provides watermark injection during inference. For detection, see [sbw](../sbw/).

## Features

- **Zero-overhead design** — Watermark computation runs in parallel with model forward pass
- **CUDAGraph compatible** — Works with vLLM's CUDAGraph optimization
- **Batch-aware** — Efficient processing of multiple sequences
- **Configurable** — Full control over watermark strength and parameters

## Installation

```bash
pip install vllm-sbw
```

This automatically installs `sbw` as a dependency.

## Quick Start

### With vLLM Server

Start vLLM with the watermark logits processor:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --logits-processors vllm_sbw:SBWLogitsProcessor
```

Then use the OpenAI-compatible API with watermark parameters:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Explain quantum computing:",
    max_tokens=200,
    extra_body={
        "vllm_xargs": {
            "gamma": 0.25,
            "delta": 2.0,
        }
    }
)

print(response.choices[0].text)
```

### With vLLM Offline

```python
from vllm import LLM, SamplingParams
from vllm_sbw import SBWLogitsProcessor

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

sampling_params = SamplingParams(
    max_tokens=200,
    temperature=1.0,
    logits_processors=[SBWLogitsProcessor],
)

outputs = llm.generate(["Explain quantum computing:"], sampling_params)
print(outputs[0].outputs[0].text)
```

## Parameters

Watermark parameters are passed via `vllm_xargs` in the API request:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gamma` | Green list fraction (0.0-1.0) | 0.25 |
| `delta` | Logit bias for green tokens | 2.0 |
| `hash_key` | Secret key (must match detector) | 15485863 |
| `seeding_scheme` | "selfhash", "lefthash", etc. | "selfhash" |

### Disabling Watermark

Set `delta=0.0` to generate text without watermarking:

```python
extra_body={"vllm_xargs": {"gamma": 0.25, "delta": 0.0}}
```

## Detecting Watermarks

Use the `sbw` package to detect watermarks in generated text:

```python
from sbw import WatermarkDetector
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
detector = WatermarkDetector(
    device=torch.device("cuda"),
    tokenizer=tokenizer,
    vocab=list(range(len(tokenizer))),
    gamma=0.25,  # Must match generation
    seeding_scheme="selfhash",
    hash_key=15485863,  # Must match generation
    z_threshold=4.0,
)

result = detector.detect(text=generated_text)
print(f"Watermark detected: {result['prediction']}")
```

## Performance

SBW is designed for minimal inference overhead:

- **Parallel computation**: Watermark logits computed during model forward pass
- **GPU-native**: All operations on GPU, no CPU round-trips
- **CUDAGraph support**: Compatible with vLLM's graph capture optimization

See the [benchmarks](../benchmarks/) directory for detailed performance measurements.

## Related Projects

- [sbw](../sbw/) — Detection library
- [waterpipe](https://github.com/si-mon-jinn/waterpipe) — Evaluation pipeline
- [flip-dont-shuffle](https://github.com/si-mon-jinn/flip-dont-shuffle) — Paper repository

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
