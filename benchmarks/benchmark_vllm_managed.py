#!/usr/bin/env python3
"""
Benchmark script that manages vLLM server lifecycle.
Runs baseline (no logits processor) then watermark (with logits processor).
Outputs JSON results.

Usage: python benchmark_vllm_managed.py [output.json] [prompt_multiplier]
"""

import sys
import os
import re
import json
import subprocess
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_FILE = os.path.join(PROJECT_ROOT, "server/config/config.yaml")
BASE_URL = "http://127.0.0.1:8008"
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]


def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(['git', 'diff', '--quiet'], stderr=subprocess.DEVNULL) != 0
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def start_server(with_logits):
    """Start vLLM server and wait for it to be ready."""
    log_file = os.path.join(PROJECT_ROOT, "server_benchmark.log")
    
    cmd = ["vllm", "serve", "--config", CONFIG_FILE]
    if with_logits:
        cmd += ["--logits-processors", "vllm_sbw:SBWLogitsProcessor"]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + ":" + env.get("PYTHONPATH", "")
    
    print(f"Starting vLLM server (logits processor: {with_logits})...")
    proc = subprocess.Popen(cmd, stdout=open(log_file, 'w'), stderr=subprocess.STDOUT, env=env)
    
    # Wait for server
    for _ in range(60):
        try:
            subprocess.check_call(['curl', '-s', f'{BASE_URL}/health'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Server ready!")
            return proc
        except subprocess.CalledProcessError:
            time.sleep(2)
    
    proc.kill()
    raise RuntimeError("Server failed to start")


def stop_server(proc):
    """Stop vLLM server."""
    print("Stopping server...")
    proc.terminate()
    proc.wait(timeout=10)
    subprocess.run(['pkill', '-f', 'vllm serve'], stderr=subprocess.DEVNULL)
    time.sleep(2)


def run_bench(batch_size, input_len, output_len, num_prompts):
    """Run vllm bench and return output."""
    cmd = [
        "vllm", "bench", "serve",
        "--model", "Qwen/Qwen3-14B",
        "--base-url", BASE_URL,
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--request-rate", "inf",
        "--max-concurrency", str(batch_size)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    return result.stdout


def parse_bench_output(output):
    """Parse vllm bench output and extract metrics."""
    def extract(pattern):
        m = re.search(pattern, output)
        return float(m.group(1)) if m else 0
    
    return {
        'token_throughput': extract(r'Output token throughput \(tok/s\):\s+([\d.]+)'),
        'tpot_mean': extract(r'Mean TPOT \(ms\):\s+([\d.]+)'),
        'tpot_median': extract(r'Median TPOT \(ms\):\s+([\d.]+)'),
        'tpot_p99': extract(r'P99 TPOT \(ms\):\s+([\d.]+)'),
        'ttft_mean': extract(r'Mean TTFT \(ms\):\s+([\d.]+)'),
    }


def main():
    default_output = os.path.join(SCRIPT_DIR, "benchmark_results", "vllm_managed.json")
    output_file = sys.argv[1] if len(sys.argv) > 1 else default_output
    prompt_multiplier = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    input_len, output_len = 128, 256
    
    git_info = get_git_info()
    results = {}
    
    print("=== vLLM Managed Benchmark ===")
    print(f"Prompt multiplier: {prompt_multiplier}x, Input: {input_len}, Output: {output_len}")
    print(f"Git: {git_info['commit'][:8]} ({'dirty' if git_info['dirty'] else 'clean'})\n")
    
    # Baseline
    print("=" * 60)
    print("BASELINE (no logits processor)")
    print("=" * 60)
    proc = start_server(with_logits=False)
    try:
        baseline = {}
        for batch in BATCH_SIZES:
            num_prompts = batch * prompt_multiplier
            print(f"\nBatch size: {batch}, num_prompts: {num_prompts}")
            output = run_bench(batch, input_len, output_len, num_prompts)
            baseline[batch] = parse_bench_output(output)
    finally:
        stop_server(proc)
    
    time.sleep(5)
    
    # Watermark
    print("\n" + "=" * 60)
    print("WATERMARK (with SBWLogitsProcessor)")
    print("=" * 60)
    proc = start_server(with_logits=True)
    try:
        watermark = {}
        for batch in BATCH_SIZES:
            num_prompts = batch * prompt_multiplier
            print(f"\nBatch size: {batch}, num_prompts: {num_prompts}")
            output = run_bench(batch, input_len, output_len, num_prompts)
            watermark[batch] = parse_bench_output(output)
    finally:
        stop_server(proc)
    
    # Build JSON results
    for batch in BATCH_SIZES:
        if baseline[batch]['tpot_mean'] > 0:
            results[str(batch)] = {
                "baseline": {
                    "per_item_ms": baseline[batch]['tpot_mean'] * output_len,
                    "per_token_ms": baseline[batch]['tpot_mean'],
                    "throughput": baseline[batch]['token_throughput'],
                    "ttft_mean": baseline[batch]['ttft_mean'],
                    "tpot_median": baseline[batch]['tpot_median'],
                    "tpot_p99": baseline[batch]['tpot_p99']
                },
                "watermark": {
                    "per_item_ms": watermark[batch]['tpot_mean'] * output_len,
                    "per_token_ms": watermark[batch]['tpot_mean'],
                    "throughput": watermark[batch]['token_throughput'],
                    "ttft_mean": watermark[batch]['ttft_mean'],
                    "tpot_median": watermark[batch]['tpot_median'],
                    "tpot_p99": watermark[batch]['tpot_p99']
                }
            }
    
    data = {
        "metadata": {
            "tool": "vllm_bench",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Qwen/Qwen3-14B",
            "dataset": f"Random (input_len={input_len}, output_len={output_len})",
            "prompt_multiplier": prompt_multiplier,
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"]
        },
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ JSON results saved to: {output_file}")
    print(f"✓ Run 'python generate_report.py {output_file}' to generate markdown report")


if __name__ == "__main__":
    main()
