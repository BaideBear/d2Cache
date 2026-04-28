#!/bin/bash
# ============================================================================
# SlidingWindowCache 公平对比测试脚本
#
# 对比方法:
#   NoCache          - 无缓存基线
#   PrefixCache      - PrefixCache (block_length=32)
#   SW_W{16,32,64}   - SlidingWindowCache (window_size=16/32/64)
#   d2Cache          - d2Cache (高级缓存基线)
#
# 使用方式:
#   bash scripts/benchmark_sw.sh                              # 默认配置
#   bash scripts/benchmark_sw.sh --model llada-base           # 指定模型
#   bash scripts/benchmark_sw.sh --datasets gsm8k,math-500    # 指定数据集
#   bash scripts/benchmark_sw.sh --quick                      # 快速测试(每数据集10条)
#   bash scripts/benchmark_sw.sh --only-sw                    # 仅测试SW不同窗口大小
# ============================================================================

set -euo pipefail

# ─── 默认配置 ────────────────────────────────────────────────────────────
MODEL="${MODEL:-llada-inst}"
DATASETS="${DATASETS:-gsm8k,humaneval}"
SEED="${SEED:-42}"
WINDOW_SIZES=(16 32 64)
BENCHMARK_DIR="benchmark_results"
QUICK_MODE=false
ONLY_SW=false

# ─── 参数解析 ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL="$2";    shift 2 ;;
        --datasets)   DATASETS="$2"; shift 2 ;;
        --quick)      QUICK_MODE=true; shift ;;
        --only-sw)    ONLY_SW=true;  shift ;;
        --seed)       SEED="$2";     shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra DATASET_ARR <<< "$DATASETS"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${BENCHMARK_DIR}/summary_${MODEL}_${TIMESTAMP}.txt"

# ─── 准备 ────────────────────────────────────────────────────────────────
mkdir -p "$BENCHMARK_DIR"
echo "============================================" | tee "$SUMMARY_FILE"
echo " SlidingWindowCache 公平对比测试"              | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"
echo "模型:     $MODEL"                               | tee -a "$SUMMARY_FILE"
echo "数据集:   ${DATASETS}"                          | tee -a "$SUMMARY_FILE"
echo "种子:     $SEED"                                | tee -a "$SUMMARY_FILE"
echo "快速模式: $QUICK_MODE"                          | tee -a "$SUMMARY_FILE"
echo "仅SW:     $ONLY_SW"                             | tee -a "$SUMMARY_FILE"
echo "时间:     $TIMESTAMP"                           | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ─── 构建测试配置列表 ────────────────────────────────────────────────────
declare -A CONFIG_NAMES
declare -A CONFIG_ARGS

if $ONLY_SW; then
    for W in "${WINDOW_SIZES[@]}"; do
        key="SW_W${W}"
        CONFIG_NAMES[$key]="$key"
        CONFIG_ARGS[$key]="cache=sw cache.window_size=${W}"
    done
else
    CONFIG_NAMES["NoCache"]="NoCache"
    CONFIG_ARGS["NoCache"]="cache=null"

    CONFIG_NAMES["PrefixCache"]="PrefixCache"
    CONFIG_ARGS["PrefixCache"]="cache=prefix"

    for W in "${WINDOW_SIZES[@]}"; do
        key="SW_W${W}"
        CONFIG_NAMES[$key]="$key"
        CONFIG_ARGS[$key]="cache=sw cache.window_size=${W}"
    done

    CONFIG_NAMES["d2Cache"]="d2Cache"
    CONFIG_ARGS["d2Cache"]="cache=d2cache"
fi

CONFIG_ORDER=()
for key in "${!CONFIG_NAMES[@]}"; do
    CONFIG_ORDER+=("$key")
done
# 确保顺序一致 (按配置列表原始顺序)
if $ONLY_SW; then
    CONFIG_ORDER=()
    for W in "${WINDOW_SIZES[@]}"; do
        CONFIG_ORDER+=("SW_W${W}")
    done
else
    CONFIG_ORDER=("NoCache" "PrefixCache")
    for W in "${WINDOW_SIZES[@]}"; do
        CONFIG_ORDER+=("SW_W${W}")
    done
    CONFIG_ORDER+=("d2Cache")
fi

# ─── 运行测试 ────────────────────────────────────────────────────────────
for dataset in "${DATASET_ARR[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$SUMMARY_FILE"
    echo "  数据集: $dataset"                             | tee -a "$SUMMARY_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$SUMMARY_FILE"

    for config_key in "${CONFIG_ORDER[@]}"; do
        config_name="${CONFIG_NAMES[$config_key]}"
        config_args="${CONFIG_ARGS[$config_key]}"
        output_dir="${BENCHMARK_DIR}/${MODEL}/${dataset}/${config_name}/${TIMESTAMP}"

        echo "" | tee -a "$SUMMARY_FILE"
        echo "▶ 运行: $config_name" | tee -a "$SUMMARY_FILE"
        echo "  参数: $config_args" | tee -a "$SUMMARY_FILE"
        echo "  输出: $output_dir" | tee -a "$SUMMARY_FILE"

        extra=""
        if $QUICK_MODE; then
            extra="dataset.size=10"
        fi

        # 跳过 d2Cache quick 模式 (它需要 attn_implementation=eager，影响公平性)
        if $QUICK_MODE && [[ "$config_key" == "d2Cache" ]]; then
            echo "  ⏭ 跳过 (quick 模式，d2Cache 需 eager attention)" | tee -a "$SUMMARY_FILE"
            continue
        fi

        set +e
        python eval.py \
            model="$MODEL" \
            dataset.name="$dataset" \
            seed="$SEED" \
            $config_args \
            $extra \
            hydra.run.dir="$output_dir" \
            2>&1 | tail -n 20
        exit_code=$?
        set -e

        if [ $exit_code -eq 0 ]; then
            echo "  ✅ 完成" | tee -a "$SUMMARY_FILE"
        else
            echo "  ❌ 失败 (退出码: $exit_code)" | tee -a "$SUMMARY_FILE"
        fi
    done
done

# ─── 收集结果汇总 ────────────────────────────────────────────────────────
echo "" | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"
echo " 结果汇总"                                    | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"

for dataset in "${DATASET_ARR[@]}"; do
    echo "" | tee -a "$SUMMARY_FILE"
    echo "📊 $dataset" | tee -a "$SUMMARY_FILE"

    # 表头
    printf "  %-16s | %10s | %12s | %10s | %12s | %12s\n" \
        "Method" "Accuracy" "Throughput" "TPS" "Latency(s)" "PeakMem(GB)" \
        | tee -a "$SUMMARY_FILE"
    printf "  %-16s-+-%-10s-+-%-12s-+-%-10s-+-%-12s-+-%-12s\n" \
        "----------------" "----------" "------------" "----------" "------------" "------------" \
        | tee -a "$SUMMARY_FILE"

    for config_key in "${CONFIG_ORDER[@]}"; do
        config_name="${CONFIG_NAMES[$config_key]}"

        if $QUICK_MODE && [[ "$config_key" == "d2Cache" ]]; then
            printf "  %-16s | %10s | %12s | %10s | %12s | %12s\n" \
                "$config_name" "(skipped)" "-" "-" "-" "-" \
                | tee -a "$SUMMARY_FILE"
            continue
        fi

        output_dir="${BENCHMARK_DIR}/${MODEL}/${dataset}/${config_name}/${TIMESTAMP}"
        results_file="${output_dir}/results.json"

        if [ -f "$results_file" ]; then
            # 用 python 提取关键指标
            python3 -c "
import json
with open('${results_file}') as f:
    data = json.load(f)

# 提取准确率 (根据数据集不同取不同 key)
results = data.get('results', {})
acc_keys = [k for k in results if 'acc' in k.lower() or 'pass' in k.lower() or 'exact' in k.lower() or 'f1' in k.lower()]
acc = None
if acc_keys:
    acc = results[acc_keys[0]]
    if isinstance(acc, dict) and 'acc,none' in acc:
        acc = acc['acc,none']
    if isinstance(acc, dict) and 'acc' in acc:
        acc = acc['acc']

throughput = data.get('throughput', '-')
tps = data.get('tps', '-')
latency = data.get('latency', '-')
peak_mem = data.get('peak_memory_allocated_GB', '-')

print(f'{acc}||{throughput}||{tps}||{latency}||{peak_mem}')
" 2>/dev/null | while IFS='||' read -r acc throughput tps latency peak_mem; do
                acc_fmt=$(printf "%.4f" "$acc" 2>/dev/null || echo "$acc")
                tp_fmt=$(printf "%.2f" "$throughput" 2>/dev/null || echo "$throughput")
                tps_fmt=$(printf "%.2f" "$tps" 2>/dev/null || echo "$tps")
                lat_fmt=$(printf "%.2f" "$latency" 2>/dev/null || echo "$latency")
                mem_fmt=$(printf "%.2f" "$peak_mem" 2>/dev/null || echo "$peak_mem")

                printf "  %-16s | %10s | %12s | %10s | %12s | %12s\n" \
                    "$config_name" "$acc_fmt" "$tp_fmt" "$tps_fmt" "$lat_fmt" "$mem_fmt" \
                    | tee -a "$SUMMARY_FILE"
            done
        else
            printf "  %-16s | %10s | %12s | %10s | %12s | %12s\n" \
                "$config_name" "(failed)" "-" "-" "-" "-" \
                | tee -a "$SUMMARY_FILE"
        fi
    done
done

echo "" | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"
echo " 详细结果保存在: ${BENCHMARK_DIR}/${MODEL}/" | tee -a "$SUMMARY_FILE"
echo " 汇总文件: $SUMMARY_FILE"                    | tee -a "$SUMMARY_FILE"
echo "============================================" | tee -a "$SUMMARY_FILE"
