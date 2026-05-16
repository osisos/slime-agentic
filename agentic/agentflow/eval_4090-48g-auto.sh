#!/bin/bash
# 自动化单卡双模型串行启动与评测脚本

export SGLANG_DISABLE_CUDNN_CHECK=1
export NCCL_IGNORE_CPU_AFFINITY=1
export TORCH_CUDA_ARCH_LIST="9.0"

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

RUN_TAG=${RUN_TAG:-"$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"${SCRIPT_DIR}/eval_runs/eval_4090-48g-auto_${RUN_TAG}"}
TRAJECTORY_DIR=${TRAJECTORY_DIR:-"${RUN_DIR}/trajectories"}
PLANNER_LOG=${PLANNER_LOG:-"${RUN_DIR}/planner_server.log"}
CODER_LOG=${CODER_LOG:-"${RUN_DIR}/coder_server.log"}
EVAL_LOG=${EVAL_LOG:-"${RUN_DIR}/eval_stdout_stderr.log"}

mkdir -p "${RUN_DIR}" "${TRAJECTORY_DIR}"

# ── 1. 核心黄金参数配置 ──────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"/cloud/cloud-ssd1/models/Qwen2.5-3B-Instruct"}
CODER_MODEL_PATH=${CODER_MODEL_PATH:-"/cloud/cloud-ssd1/models/Qwen3.5-4b"}

PLANNER_PORT=30000
CODER_PORT=30002

MEM_FRACTION=0.3
CTX_LEN=16384
CONCURRENCY=8
# ────────────────────────────────────────────────────────────────────────

# ── 2. 清理环境与退出机制 ────────────────────────────────────────────────
echo "🧹 [1/4] 正在清理可能残留的僵尸进程..."
pkill -9 -f sglang || true
sleep 2

# 定义退出时的清理动作：当脚本结束或被Ctrl+C时，自动杀掉后台的模型进程
cleanup() {
    echo -e "\n🛑 脚本终止，正在关闭后台运行的模型服务..."
    kill $PLANNER_PID $CODER_PID 2>/dev/null || true
    echo "✅ 清理完毕！"
}
trap cleanup EXIT INT TERM

# ── 3. 辅助函数：等待服务就绪 ────────────────────────────────────────────
wait_for_server() {
    local port=$1
    local name=$2
    echo "⏳ 等待 $name (端口 $port) 启动就绪，这可能需要一两分钟..."
    # 轮询健康检查接口，直到返回 HTTP 200
    while [ "$(curl -s -o /dev/null -w ''%{http_code}'' http://127.0.0.1:${port}/health)" != "200" ]; do
        sleep 5
    done
    echo "🚀 $name 启动成功并准备就绪！"
}

# ── 4. 串行启动后端服务 ──────────────────────────────────────────────────
echo "📦 [2/4] 正在后台启动 Planner 模型 (3B)..."
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$PLANNER_PORT" \
  --tp 1 \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CTX_LEN" \
  --disable-cuda-graph \
  --max-prefill-tokens "$CTX_LEN" \
  --max-running-requests "$CONCURRENCY" \
  --trust-remote-code > "$PLANNER_LOG" 2>&1 &
PLANNER_PID=$!

wait_for_server "$PLANNER_PORT" "Planner"
echo "   (Planner 日志已输出至 ${PLANNER_LOG})"


echo "📦 [3/4] 正在后台启动 Coder 模型 (4B)..."
python3 -m sglang.launch_server \
  --model-path "$CODER_MODEL_PATH" \
  --port "$CODER_PORT" \
  --tp 1 \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CTX_LEN" \
  --disable-cuda-graph \
  --max-prefill-tokens "$CTX_LEN" \
  --max-running-requests "$CONCURRENCY" \
  --trust-remote-code > "$CODER_LOG" 2>&1 &
CODER_PID=$!

wait_for_server "$CODER_PORT" "Coder"
echo "   (Coder 日志已输出至 ${CODER_LOG})"

# ── 5. 运行评测脚本 ──────────────────────────────────────────────────────
echo "🎯 [4/4] 所有后端服务已就绪，开始运行 AIME 评测..."

# 这里直接调用你的评测 Python 脚本
# 请确保下面这些评测用的参数符合你的原始需求
TOKENIZER_PATH="${MODEL_PATH}"
EVAL_DATA=(aime /cloud/cloud-ssd1/data/aime-2024/aime-2024.jsonl)
OUTPUT=${OUTPUT:-"${RUN_DIR}/eval_4090-48g-auto_results.json"}
MAX_STEPS=5
TEMPERATURE=0.7
TOP_P=0.95
MAX_NEW_TOKENS=4096
SAMPLES_PER_PROMPT=8
# 只评测数据集中的某一个题目；空字符串表示评测全部。下标从 0 开始。
IDX=${IDX:-""}

export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"

echo "   (本次运行目录：${RUN_DIR})"
echo "   (轨迹目录：${TRAJECTORY_DIR})"
echo "   (评测日志：${EVAL_LOG})"
if [ -n "${IDX}" ]; then
    echo "   (只评测单题 idx=${IDX})"
fi

PY_ARGS=(
    --tokenizer "$TOKENIZER_PATH"
    --eval-data "${EVAL_DATA[@]}"
    --input-key prompt
    --label-key label
    --output "$OUTPUT"
    --concurrency "$CONCURRENCY"
    --max-steps "$MAX_STEPS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --samples-per-prompt "$SAMPLES_PER_PROMPT"
    --trajectory-dir "$TRAJECTORY_DIR"
    --planner-url "http://127.0.0.1:${PLANNER_PORT}/generate"
    --coder-url "http://127.0.0.1:${CODER_PORT}/generate"
)

if [ -n "${IDX}" ]; then
    PY_ARGS+=(--idx "$IDX")
fi

python3 "${SCRIPT_DIR}/eval_agentflow.py" "${PY_ARGS[@]}" 2>&1 | tee "$EVAL_LOG"

echo "🎉 评测全部完成！结果已保存至：$OUTPUT"
echo "🧾 轨迹与 rewarder 审计输出已保存至：$TRAJECTORY_DIR"
# 脚本运行到最后，会自动触发 trap 的 cleanup 函数，优雅关闭两个模型。
