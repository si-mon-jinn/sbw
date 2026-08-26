#!/usr/bin/env python3
"""
Benchmark script for profiling SBWLogitsProcessor with synthetic data.
Varies batch size and watermark scheme to measure performance characteristics.

Usage:
    python profile_logits_processor.py
    python profile_logits_processor.py --schemes simple_1,minhash,selfhash
    python profile_logits_processor.py --schemes minhash --context-widths 2,4,8
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from vllm_sbw import SBWLogitsProcessor
from sbw.prf import seeding_scheme_lookup
from benchmarks.bench_utils import add_benchmark_args, parse_benchmark_args


def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(['git', 'diff', '--quiet'], stderr=subprocess.DEVNULL) != 0
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


class MockVllmConfig:
    def __init__(self, vocab_size, model_name="Qwen/Qwen3-0.6B"):
        self.model_config = self
        self._vocab_size = vocab_size
        self.model = model_name

    def get_vocab_size(self):
        return self._vocab_size


def create_processor(batch_size, vocab_size, device, profile, prompt_len):
    """Create a SBWLogitsProcessor with synthetic batch data."""
    config = MockVllmConfig(vocab_size)
    processor = SBWLogitsProcessor(
        vllm_config=config, device=device, is_pin_memory=False, profile=profile,
    )
    processor.batch_prompts = []
    processor.batch_outputs = []
    processor._gammas = []
    processor._deltas = []
    for i in range(batch_size):
        processor.batch_prompts.append(
            list(range(100 + i * prompt_len, 100 + (i + 1) * prompt_len))
        )
        processor.batch_outputs.append([200 + i])
        processor._gammas.append(0.5)
        processor._deltas.append(2.0)
    return processor


def reset_profile_stats(processor):
    """Reset profiling counters on both processor and watermark."""
    if processor.profile:
        processor.profile_stats = {"collect_tokens": [], "apply_bias": []}
    wm = processor.watermark
    if wm.profile:
        wm.profile_stats = {k: [] for k in wm.profile_stats}


def benchmark_one(batch_sizes, vocab_size, config, num_iterations, warmup, device, profile, output_len=1, kernel_only=False):
    """Benchmark a single watermark config across batch sizes.
    
    If kernel_only=True, benchmarks just the watermark kernel without context collection overhead.
    """
    # Determine prompt length from config's context width
    scheme = config["seeding_scheme"]
    _, cw, _, _, _ = seeding_scheme_lookup(scheme)
    prompt_len = max(cw, 1)

    # Pre-create processors for all batch sizes to trigger torch.compile once per shape
    processors = {}
    for batch_size in batch_sizes:
        processor = create_processor(batch_size, vocab_size, device, profile, prompt_len)
        processor.watermark.gamma = config["gamma"]
        processor.watermark.delta = config["delta"]
        processor.watermark.set_seeding_scheme(scheme, hash_key=config["hash_key"])
        processor._gammas = [config["gamma"]] * batch_size
        processor._deltas = [config["delta"]] * batch_size
        processors[batch_size] = processor

    # Pre-warm all batch sizes to trigger compilation for each shape
    for batch_size in batch_sizes:
        processor = processors[batch_size]
        wm = processor.watermark
        context = torch.randint(0, 1000, (batch_size, cw), device=device)
        gamma_vec = torch.tensor([config["gamma"]] * batch_size, device=device)
        delta_vec = torch.tensor([config["delta"]] * batch_size, device=device, dtype=torch.float32)
        
        for _ in range(warmup):
            logits = torch.randn(batch_size, vocab_size, device=device)
            if kernel_only:
                # Warmup the kernel directly
                if scheme.startswith("gpu-") and wm.self_salt:
                    if wm.is_fused:
                        wm.apply_watermark_selfsalt_fused(context, logits, gamma_vec, delta_vec)
                    else:
                        wm.apply_watermark_selfsalt_direct(context, logits, gamma_vec, delta_vec)
                elif scheme.startswith("gpu-"):
                    wm.apply_watermark_fused(context, logits, gamma_vec, delta_vec)
            else:
                processor.apply(logits)
    if device.type == "cuda":
        torch.cuda.synchronize()

    results = {}
    for batch_size in batch_sizes:
        processor = processors[batch_size]

        # Reset output lists for the timed run
        for i in range(len(processor.batch_outputs)):
            processor.batch_outputs[i] = processor.batch_outputs[i][:1]

        reset_profile_stats(processor)

        # Benchmark
        times = []
        
        if kernel_only:
            # Benchmark just the watermark kernel
            wm = processor.watermark
            context = torch.randint(0, 1000, (batch_size, cw), device=device)
            gamma_vec = torch.tensor([config["gamma"]] * batch_size, device=device)
            delta_vec = torch.tensor([config["delta"]] * batch_size, device=device, dtype=torch.float32)
            
            for _ in range(num_iterations):
                logits = torch.randn(batch_size, vocab_size, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                if scheme.startswith("gpu-") and wm.self_salt:
                    if wm.is_fused:
                        wm.apply_watermark_selfsalt_fused(context, logits, gamma_vec, delta_vec)
                    else:
                        wm.apply_watermark_selfsalt_direct(context, logits, gamma_vec, delta_vec)
                elif scheme.startswith("gpu-"):
                    wm.apply_watermark_fused(context, logits, gamma_vec, delta_vec)
                else:
                    wm.get_greenlist_masks(context, logits)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        else:
            # Benchmark full apply() path
            for _ in range(num_iterations):
                logits = torch.randn(batch_size, vocab_size, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                processor.apply(logits)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                # Simulate decode: grow output like vLLM does
                if output_len > 1:
                    for o in processor.batch_outputs:
                        o.append(300)

        result = {
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "median_ms": statistics.median(times),
        }

        if profile:
            result["profile"] = processor.watermark.get_profile_summary()
            for key in ("collect_tokens", "apply_bias"):
                vals = processor.profile_stats[key]
                if vals:
                    result[key] = {"mean_ms": statistics.mean(vals),
                                   "std_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0}

        results[batch_size] = result
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile SBWLogitsProcessor with synthetic data")
    parser.add_argument("--output", type=str,
                        default=os.path.join(SCRIPT_DIR, "benchmark_results", "profile_logits_processor.json"))
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32,64")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-len", type=int, default=1,
                        help="Simulated output length: append a token after each step (default: 1 = static)")
    parser.add_argument("--kernel-only", action="store_true",
                        help="Benchmark just the watermark kernel, excluding context collection overhead")
    add_benchmark_args(parser)
    args = parser.parse_args()

    configs = parse_benchmark_args(args)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print(f"Kernel only: {args.kernel_only}")
    print(f"Configs: {[c['label'] for c in configs]}")
    print("=" * 80)

    # Baseline (delta=0)
    print("\nRunning baseline (delta=0)...")
    baseline_cfg = {"seeding_scheme": "simple_1", "gamma": 0.5, "delta": 0.0, "hash_key": 15485863}
    baseline_results = benchmark_one(batch_sizes, args.vocab_size, baseline_cfg,
                                     args.iterations, args.warmup, device, profile=False,
                                     output_len=args.output_len, kernel_only=args.kernel_only)

    # Each config
    all_results = {}
    for cfg in configs:
        print(f"\nRunning {cfg['label']}...")
        all_results[cfg["label"]] = benchmark_one(batch_sizes, args.vocab_size, cfg,
                                                   args.iterations, args.warmup, device, profile=True,
                                                   output_len=args.output_len, kernel_only=args.kernel_only)

    # Console output
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for label, wm_results in all_results.items():
        print(f"\n--- {label} ---")
        b1_ms = wm_results[batch_sizes[0]]["mean_ms"]
        for bs in batch_sizes:
            bl = baseline_results[bs]
            wm = wm_results[bs]
            overhead = wm["mean_ms"] - bl["mean_ms"]
            per_seq = wm["mean_ms"] / bs
            speedup = b1_ms / per_seq if per_seq > 0 else 0
            print(f"  B={bs:<4}  baseline={bl['mean_ms']:>7.2f}ms  watermark={wm['mean_ms']:>7.2f}ms  "
                  f"overhead={overhead:>+7.2f}ms  per_seq={per_seq:>7.2f}ms  speedup={speedup:.2f}x")

    # JSON output
    json_results = {}
    for bs in batch_sizes:
        entry = {"baseline": {"per_item_ms": baseline_results[bs]["mean_ms"],
                               "per_token_ms": baseline_results[bs]["mean_ms"] / bs}}
        for label, wm_results in all_results.items():
            wm = wm_results[bs]
            entry[label] = {"per_item_ms": wm["mean_ms"], "per_token_ms": wm["mean_ms"] / bs}
            if "profile" in wm:
                entry[label]["profile"] = wm["profile"]
        json_results[str(bs)] = entry

    data = {
        "metadata": {
            "tool": "profile_logits_processor",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "synthetic",
            "vocab_size": args.vocab_size,
            "num_iterations": args.iterations,
            "configs": configs,
            "git_commit": get_git_info()["commit"],
            "git_dirty": get_git_info()["dirty"],
        },
        "results": json_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ JSON results saved to: {args.output}")


if __name__ == "__main__":
    main()
