#!/bin/bash
# ============================================================================
# SlidingWindowCache 快速验证脚本
#
# 在 HumanEval 上限制 10 条数据，快速验证 SlidingWindowCache
# 的三种窗口大小 (16/32/64) 与 NoCache/PrefixCache 的对比。
#
# 运行:
#   bash scripts/benchmark_sw_quick.sh
#   bash scripts/benchmark_sw_quick.sh --model llada-base
# ============================================================================

set -euo pipefail

MODEL="${MODEL:-llada-inst}"
DATASET="${DATASET:-humaneval}"
SEED=42
SAMPLE_SIZE=10
BENCHMARK_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " SlidingWindowCache 快速验证"
echo "============================================"
echo "模型:       $MODEL"
echo "数据集:     $DATASET"
echo "样本数:     $SAMPLE_SIZE"
echo "============================================"

declare -A RESULTS

run_test() {
    local name="$1"
    local args="$2"
    local output_dir="${BENCHMARK_DIR}/quick/${MODEL}/${name}/${TIMESTAMP}"

    echo ""
    echo "▶ [$name]"
    echo "  args: $args"
    echo "  output: $output_dir"

    python eval.py \
        model="$MODEL" \
        dataset.name="$DATASET" \
        dataset.size="$SAMPLE_SIZE" \
        seed="$SEED" \
        $args \
        hydra.run.dir="$output_dir" \
        2>&1 | tail -n 10

    local results_file="${output_dir}/results.json"
    if [ -f "$results_file" ]; then
        python3 -c "
import json
with open('${results_file}') as f:
    data = json.load(f)

results = data.get('results', {})
acc_keys = [k for k in results if 'acc' in k.lower() or 'pass' in k.lower() or 'exact' in k.lower()]
acc = 'N/A'
if acc_keys:
    acc_val = results[acc_keys[0]]
    if isinstance(acc_val, dict) and 'acc,none' in acc_val:
        acc_val = acc_val['acc,none']
    if isinstance(acc_val, dict) and 'acc' in acc_val:
        acc_val = acc_val['acc']
    acc = f'{acc_val:.4f}' if isinstance(acc_val, (int,float)) else str(acc_val)

throughput = data.get('throughput', 'N/A')
throughput = f'{throughput:.2f}' if isinstance(throughput, (int,float)) else str(throughput)
latency = data.get('latency', 'N/A')
latency = f'{latency:.2f}' if isinstance(latency, (int,float)) else str(latency)

print(f'OK|{acc}|{throughput}|{latency}')
" 2>/dev/null || echo "FAIL"
    else
        echo "FAIL(results.json not found)"
    fi
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 基线对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_test "NoCache"        "cache=null"
run_test "PrefixCache"    "cache=prefix"
run_test "SW_W16"         "cache=sw cache.window_size=16"
run_test "SW_W32"         "cache=sw cache.window_size=32"
run_test "SW_W64"         "cache=sw cache.window_size=64"

echo ""
echo "============================================"
echo " 验证完成"
echo " 结果目录: ${BENCHMARK_DIR}/quick/${MODEL}/"
echo "============================================"
