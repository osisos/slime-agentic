#!/bin/bash
# AgentFlow local top-8 evaluation script.
# Uses one local model/server for planner, coder, verifier, and rewarder.
# Draws 8 independent samples per prompt with temperature=0.7 and reports
# the probability that at least one of the 8 samples completes the task.

set -e

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH=${MODEL_PATH:-"/data/AgentFlow_pro-Qwen25-7B-RL/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/data/models/qwen25_7b"}

EVAL_DATA=(
    aime /data/aime-2024/aime-2024.jsonl
)

OUTPUT=${OUTPUT:-"$(dirname "$0")/eval_local_top8_results.json"}
TRAJECTORY_DIR=${TRAJECTORY_DIR:-""}

TP=${TP:-1}
MEM_FRACTION=${MEM_FRACTION:-0.7}
CTX_LEN=${CTX_LEN:-32768}
CONCURRENCY=${CONCURRENCY:-16}
MAX_STEPS=${MAX_STEPS:-5}

TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-8}

# Debug limit for prompts, not total attempts. 0 = no limit.
NUM_SAMPLES=${NUM_SAMPLES:-0}

# Run only one sample by zero-based dataset index. Empty = all samples.
IDX=${IDX:-""}

PLANNER_PORT=${PLANNER_PORT:-30000}
CODER_PORT=${PLANNER_PORT}
AUTO_START=${AUTO_START:-1}

# ── Environment ───────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"

# ── Python args ───────────────────────────────────────────────────────────────

PY_ARGS=(
    --tokenizer  "${TOKENIZER_PATH}"
    --eval-data  "${EVAL_DATA[@]}"
    --input-key  prompt
    --label-key  label
    --output     "${OUTPUT}"
    --concurrency "${CONCURRENCY}"
    --max-steps  "${MAX_STEPS}"
    --temperature "${TEMPERATURE}"
    --top-p       "${TOP_P}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --samples-per-prompt "${SAMPLES_PER_PROMPT}"
    --tp          "${TP}"
    --mem-fraction "${MEM_FRACTION}"
    --ctx-len     "${CTX_LEN}"
    --planner-port  "${PLANNER_PORT}"
    --coder-port    "${CODER_PORT}"
)

if [ "${AUTO_START}" = "1" ]; then
    PY_ARGS+=(--model "${MODEL_PATH}" --start-servers)
else
    PY_ARGS+=(
        --planner-url  "http://127.0.0.1:${PLANNER_PORT}/generate"
        --coder-url    "http://127.0.0.1:${CODER_PORT}/generate"
    )
fi

if [ -n "${TRAJECTORY_DIR}" ]; then
    PY_ARGS+=(--trajectory-dir "${TRAJECTORY_DIR}")
fi

if [ "${NUM_SAMPLES}" -gt 0 ] 2>/dev/null; then
    PY_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

if [ -n "${IDX}" ]; then
    PY_ARGS+=(--idx "${IDX}")
fi

if [ "${AUTO_START}" != "1" ]; then
    echo "============================================================"
    echo " 手动模式：请确保同一个 SGLang 服务器已在运行："
    echo "   Local/Planner/Coder 服务器 : port ${PLANNER_PORT}"
    echo "   coder_port 将复用 planner_port: ${CODER_PORT}"
    echo ""
    echo " 快速启动示例："
    echo "   python -m sglang.launch_server \\"
    echo "     --model-path ${MODEL_PATH} --port ${PLANNER_PORT} \\"
    echo "     --tp ${TP} --mem-fraction-static ${MEM_FRACTION} \\"
    echo "     --context-length ${CTX_LEN} --trust-remote-code &"
    echo "============================================================"
    echo ""
fi

# ── Run ───────────────────────────────────────────────────────────────────────

echo "▶ 开始 local top-${SAMPLES_PER_PROMPT} 评估..."
echo "  模型       : ${MODEL_PATH}"
echo "  Tokenizer  : ${TOKENIZER_PATH}"
echo "  输出文件   : ${OUTPUT}"
echo "  TP         : ${TP}"
echo "  Planner端口: ${PLANNER_PORT}"
echo "  Coder端口  : ${CODER_PORT}"
echo "  并发数     : ${CONCURRENCY}"
echo "  最大步数   : ${MAX_STEPS}"
echo "  温度       : ${TEMPERATURE}"
echo "  每题采样数 : ${SAMPLES_PER_PROMPT}"
if [ -n "${IDX}" ]; then
    echo "  单条样本   : ${IDX}"
fi
echo ""

python3 "${SCRIPT_DIR}/eval_agentflow.py" "${PY_ARGS[@]}"

echo ""
echo "✓ local top-${SAMPLES_PER_PROMPT} 评估完成，结果保存至：${OUTPUT}"
