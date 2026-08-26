#!/bin/bash
# Run all three benchmarks for watermark overhead analysis.
#
# Usage:
#   ./run_all_benchmarks.sh                          # defaults: GPU 0, simple_1, batch up to 128
#   ./run_all_benchmarks.sh --gpu 1                  # use GPU 1
#   ./run_all_benchmarks.sh --schemes simple_1,minhash
#   ./run_all_benchmarks.sh --suffix a100            # output files: *_a100.json
#   ./run_all_benchmarks.sh --venv /path/to/venv     # custom venv path
#
# Requires: vLLM venv with dependencies installed, server/config/config.yaml configured.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
GPU=0
SCHEMES="simple_1"
BATCH_SIZES="1,4,8,16,32,64,128"
SUFFIX=""
VENV="${PROJECT_ROOT}/../vllm-watermarking/vllm_env"
MULTIPLIER=10

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)        GPU="$2"; shift 2 ;;
        --schemes)    SCHEMES="$2"; shift 2 ;;
        --batches)    BATCH_SIZES="$2"; shift 2 ;;
        --suffix)     SUFFIX="_$2"; shift 2 ;;
        --venv)       VENV="$2"; shift 2 ;;
        --multiplier) MULTIPLIER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

OUTDIR="$SCRIPT_DIR/benchmark_results"
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
source "$VENV/bin/activate"

echo "============================================"
echo "Watermark Benchmark Suite"
echo "============================================"
echo "GPU:          $GPU"
echo "Schemes:      $SCHEMES"
echo "Batch sizes:  $BATCH_SIZES"
echo "Multiplier:   ${MULTIPLIER}x"
echo "Suffix:       ${SUFFIX:-none}"
echo "Venv:         $VENV"
echo "Output dir:   $OUTDIR"
echo "============================================"
echo ""

# --- Benchmark 1: Synthetic profiling (no server) ---
echo "[1/3] Synthetic profiling..."
python "$SCRIPT_DIR/profile_logits_processor.py" \
    --schemes "$SCHEMES" \
    --batch-sizes "$BATCH_SIZES" \
    --output "$OUTDIR/profile_logits_processor${SUFFIX}.json"
echo ""

# --- Benchmark 2: API burst (needs server) ---
echo "[2/3] API burst benchmark..."
echo "Starting server with watermark..."
vllm serve --config "$PROJECT_ROOT/server/config/config.yaml" \
    --logits-processors vllm_sbw:SBWLogitsProcessor \
    > /tmp/vllm_bench_server.log 2>&1 &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 90); do
    if curl -s http://127.0.0.1:8008/health > /dev/null 2>&1; then
        echo "Server ready"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died. Check /tmp/vllm_bench_server.log"
        exit 1
    fi
    sleep 2
done

if ! curl -s http://127.0.0.1:8008/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within timeout"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

python "$SCRIPT_DIR/benchmark_batch_impact_custom.py" \
    --schemes "$SCHEMES" \
    --batch-sizes "$BATCH_SIZES" \
    --output "$OUTDIR/custom_api${SUFFIX}.json"

echo "Stopping server..."
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
sleep 5
echo ""

# --- Benchmark 3: vLLM managed (manages own server) ---
echo "[3/3] vLLM managed benchmark..."
python "$SCRIPT_DIR/benchmark_vllm_managed.py" \
    "$OUTDIR/vllm_managed${SUFFIX}.json" \
    "$MULTIPLIER"
echo ""

# --- Generate reports ---
echo "Generating reports..."
python "$SCRIPT_DIR/generate_report.py" "$OUTDIR/profile_logits_processor${SUFFIX}.json"
python "$SCRIPT_DIR/generate_report.py" "$OUTDIR/custom_api${SUFFIX}.json"
python "$SCRIPT_DIR/generate_report.py" "$OUTDIR/vllm_managed${SUFFIX}.json"

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "Results in: $OUTDIR/*${SUFFIX}.*"
echo "============================================"
