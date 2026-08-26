#!/usr/bin/env python3
"""
Generate markdown report from benchmark JSON results.
Supports both old single-watermark format and new multi-config format.

Usage: python generate_report.py [results.json] [output.md]
"""

import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")


def detect_config_labels(results):
    """Detect watermark config labels from results (all keys except 'baseline')."""
    sample = results[list(results.keys())[0]]
    return [k for k in sample.keys() if k != "baseline"]


def generate_report(results_file, output_file):
    with open(results_file) as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]
    batch_sizes = sorted([int(k) for k in results.keys()])
    labels = detect_config_labels(results)

    # Handle old format (single "watermark" key)
    is_old_format = labels == ["watermark"]

    git_commit = metadata.get("git_commit", "N/A")
    git_dirty = metadata.get("git_dirty")
    git_status = f"{git_commit[:8]}" if git_commit != "N/A" else "N/A"
    if git_dirty is True:
        git_status += " (dirty)"
    elif git_dirty is False:
        git_status += " (clean)"

    configs_info = ""
    if "configs" in metadata:
        configs_info = f"**Configs:** {', '.join(c['label'] for c in metadata['configs'])}  \n"

    md = f"""# Watermark Benchmark Analysis

**Date:** {metadata.get('date', 'N/A')}  
**Model:** {metadata.get('model', 'N/A')}  
**Tool:** {metadata.get('tool', 'N/A')}  
{configs_info}**Git:** {git_status}

## Per-Token Cost (ms/token)

| Batch Size | Baseline |"""

    for label in labels:
        md += f" {label} |"
    md += "\n|------------|----------|"
    for _ in labels:
        md += "----------|"
    md += "\n"

    for bs in batch_sizes:
        r = results[str(bs)]
        row = f"| {bs:<10} | {r['baseline']['per_token_ms']:>8.3f} |"
        for label in labels:
            row += f" {r[label]['per_token_ms']:>8.3f} |"
        md += row + "\n"

    md += "\n## Overhead vs Baseline\n\n"
    md += "| Batch Size |"
    for label in labels:
        md += f" {label} (ms) | {label} (%) |"
    md += "\n|------------|"
    for _ in labels:
        md += "----------|----------|"
    md += "\n"

    for bs in batch_sizes:
        r = results[str(bs)]
        bl = r["baseline"]["per_token_ms"]
        row = f"| {bs:<10} |"
        for label in labels:
            wm = r[label]["per_token_ms"]
            overhead = wm - bl
            rel = (overhead / bl) * 100 if bl > 0 else 0
            row += f" {overhead:>+8.3f} | {rel:>+8.1f} |"
        md += row + "\n"

    if len(labels) > 1:
        md += "\n## Cross-Scheme Comparison\n\n"
        md += "Relative overhead at largest batch size "
        max_bs = batch_sizes[-1]
        r = results[str(max_bs)]
        bl = r["baseline"]["per_token_ms"]
        md += f"(B={max_bs}):\n\n"
        md += "| Scheme | Per-Token (ms) | Overhead (ms) | Overhead (%) |\n"
        md += "|--------|---------------|--------------|-------------|\n"
        for label in labels:
            wm = r[label]["per_token_ms"]
            overhead = wm - bl
            rel = (overhead / bl) * 100 if bl > 0 else 0
            md += f"| {label} | {wm:.3f} | {overhead:+.3f} | {rel:+.1f} |\n"

    with open(output_file, "w") as f:
        f.write(md)
    print(f"✓ Report saved to: {output_file}")

    # Console summary
    print(f"\n{'Batch':<8} {'Baseline':>10}", end="")
    for label in labels:
        print(f" {label:>15}", end="")
    print()
    print("-" * (18 + 15 * len(labels)))
    for bs in batch_sizes:
        r = results[str(bs)]
        print(f"{bs:<8} {r['baseline']['per_token_ms']:>10.3f}", end="")
        for label in labels:
            print(f" {r[label]['per_token_ms']:>15.3f}", end="")
        print()
    print("(per-token ms)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        json_files = [f for f in os.listdir(DEFAULT_RESULTS_DIR) if f.endswith(".json")]
        if json_files:
            results_file = os.path.join(DEFAULT_RESULTS_DIR, sorted(json_files)[-1])
            print(f"Using default: {results_file}")
        else:
            print(f"Usage: python generate_report.py [results.json] [output.md]")
            print(f"No JSON files found in {DEFAULT_RESULTS_DIR}")
            sys.exit(1)
    else:
        results_file = sys.argv[1]

    output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(results_file)[0] + "_report.md"
    generate_report(results_file, output_file)
