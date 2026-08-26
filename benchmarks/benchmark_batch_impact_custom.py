#!/usr/bin/env python3
"""
Benchmark script to measure watermark batch processing with vLLM server.
Sends synchronized bursts of requests to force specific batch sizes.
Supports multi-scheme sweeps via CLI arguments.

Usage:
    python benchmark_batch_impact_custom.py
    python benchmark_batch_impact_custom.py --schemes simple_1,minhash,selfhash
    python benchmark_batch_impact_custom.py --schemes minhash --context-widths 2,4,8

Requires: Running vLLM server with watermark logits processor.
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

from openai import AsyncOpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from benchmarks.bench_utils import add_benchmark_args, parse_benchmark_args


def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(['git', 'diff', '--quiet'], stderr=subprocess.DEVNULL) != 0
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


async def generate_burst(client, batch_size, config, max_tokens=100):
    """Send a burst of requests simultaneously with given watermark config."""
    prompts = [f"Write a short paragraph about topic {i}." for i in range(batch_size)]

    start = time.time()
    tasks = [
        client.completions.create(
            model="Qwen/Qwen3-14B",
            prompt=p,
            max_tokens=max_tokens,
            temperature=0.7,
            extra_body={"vllm_xargs": {
                "gamma": config["gamma"],
                "delta": config["delta"],
                "seeding_scheme": config["seeding_scheme"],
                "hash_key": config["hash_key"],
            }},
        )
        for p in prompts
    ]
    responses = await asyncio.gather(*tasks)
    total_time = (time.time() - start) * 1000

    tokens_per_request = [r.usage.completion_tokens for r in responses]
    return {
        "batch_size": batch_size,
        "total_time_ms": total_time,
        "tokens_per_request": tokens_per_request,
        "total_tokens": sum(tokens_per_request),
        "mean_tokens": statistics.mean(tokens_per_request),
    }


async def run_batch_size_sweep(client, batch_sizes, config, num_iterations=5, max_tokens=100):
    """Run multiple iterations for each batch size."""
    label = config.get("label", config["seeding_scheme"])
    print(f"\nRunning sweep: {label}")
    print("-" * 60)

    results = {}
    for batch_size in batch_sizes:
        print(f"  Batch size: {batch_size}", end="", flush=True)
        iteration_results = []
        for _ in range(num_iterations):
            result = await generate_burst(client, batch_size, config, max_tokens)
            iteration_results.append(result)
            await asyncio.sleep(0.5)

        total_times = [r["total_time_ms"] for r in iteration_results]
        results[batch_size] = {
            "mean_total_time_ms": statistics.mean(total_times),
            "std_total_time_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
            "mean_tokens_per_req": statistics.mean([r["mean_tokens"] for r in iteration_results]),
        }
        print(f"  {results[batch_size]['mean_total_time_ms']:.2f} ± {results[batch_size]['std_total_time_ms']:.2f}ms")

    return results


def calculate_metrics(baseline_results, config_results_map, batch_sizes):
    """Calculate per-item and per-token metrics for all configs."""
    metrics = {}
    for bs in batch_sizes:
        bl = baseline_results[bs]
        bl_per_item = bl["mean_total_time_ms"] / bs
        bl_total_tokens = bl["mean_tokens_per_req"] * bs
        bl_per_token = bl["mean_total_time_ms"] / bl_total_tokens if bl_total_tokens > 0 else 0

        entry = {"baseline": {
            "per_item_ms": bl_per_item, "per_token_ms": bl_per_token,
            "mean_total_time_ms": bl["mean_total_time_ms"], "std_total_time_ms": bl["std_total_time_ms"],
        }}

        for label, wm_results in config_results_map.items():
            wm = wm_results[bs]
            wm_per_item = wm["mean_total_time_ms"] / bs
            wm_total_tokens = wm["mean_tokens_per_req"] * bs
            wm_per_token = wm["mean_total_time_ms"] / wm_total_tokens if wm_total_tokens > 0 else 0
            entry[label] = {
                "per_item_ms": wm_per_item, "per_token_ms": wm_per_token,
                "mean_total_time_ms": wm["mean_total_time_ms"], "std_total_time_ms": wm["std_total_time_ms"],
            }

        metrics[str(bs)] = entry
    return metrics


async def main():
    parser = argparse.ArgumentParser(description="API-level watermark benchmark with request bursts")
    parser.add_argument("--output", type=str,
                        default=os.path.join(SCRIPT_DIR, "benchmark_results", "custom_api.json"))
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8008/v1")
    add_benchmark_args(parser)
    args = parser.parse_args()

    configs = parse_benchmark_args(args)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    client = AsyncOpenAI(api_key="dummy", base_url=args.base_url)

    print("=== Watermark Batch Processing Benchmark ===")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Iterations: {args.iterations}, Max tokens: {args.max_tokens}")
    print(f"Configs: {[c['label'] for c in configs]}")

    # Baseline (delta=0)
    print("\n" + "=" * 60)
    print("BASELINE (delta=0)")
    print("=" * 60)
    baseline_cfg = {"seeding_scheme": "simple_1", "gamma": 0.5, "delta": 0.0, "hash_key": 15485863, "label": "baseline"}
    baseline_results = await run_batch_size_sweep(client, batch_sizes, baseline_cfg, args.iterations, args.max_tokens)

    # Each config
    config_results_map = {}
    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"WATERMARK: {cfg['label']}")
        print("=" * 60)
        config_results_map[cfg["label"]] = await run_batch_size_sweep(
            client, batch_sizes, cfg, args.iterations, args.max_tokens)

    # Metrics
    metrics = calculate_metrics(baseline_results, config_results_map, batch_sizes)

    # Console summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Batch':<8} {'Baseline':>10}"
    for label in config_results_map:
        header += f" {label:>15}"
    print(header)
    print("-" * len(header))
    for bs in batch_sizes:
        row = f"{bs:<8} {metrics[str(bs)]['baseline']['per_token_ms']:>10.3f}"
        for label in config_results_map:
            row += f" {metrics[str(bs)][label]['per_token_ms']:>15.3f}"
        print(row)
    print("(per-token ms)")

    # JSON output
    git_info = get_git_info()
    data = {
        "metadata": {
            "tool": "custom_api",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Qwen/Qwen3-14B",
            "dataset": f"Synchronized request bursts ({args.iterations} iterations)",
            "max_tokens": args.max_tokens,
            "num_iterations": args.iterations,
            "configs": configs,
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
        },
        "results": metrics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ JSON results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
