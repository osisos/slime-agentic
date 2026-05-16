  #!/bin/bash
  # AgentFlow 32G top-8 evaluation script.
  # Follows eval_local_top8.sh and keeps the 32G model/memory defaults.

  set -e

  # ── Config ───────────────────────────────────────────────────────────────────

  export SGLANG_DISABLE_CUDNN_CHECK=1
  export NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY:-1}
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0"}

  MODEL_PATH=${MODEL_PATH:-"/cloud/cloud-ssd1/models/Qwen2.5-3B-Instruct"}
  CODER_MODEL_PATH=${CODER_MODEL_PATH:-"/cloud/cloud-ssd1/models/Qwen3.5-4b"}
  TOKENIZER_PATH=${TOKENIZER_PATH:-"${MODEL_PATH}"}

  EVAL_DATA=(
      aime /cloud/cloud-ssd1/data/aime-2024/aime-2024.jsonl
  )

  OUTPUT=${OUTPUT:-"$(dirname "$0")/eval_32g_top8_results.json"}
  TRAJECTORY_DIR=${TRAJECTORY_DIR:-""}

  TP=${TP:-1}
  # ✅ 核心修改点：将默认的 0.35 降低到 0.15，确保双模型 KV Cache 不会撑爆显存
  MEM_FRACTION=${MEM_FRACTION:-0.25} 
  CTX_LEN=${CTX_LEN:-s}
  CONCURRENCY=${CONCURRENCY:-4}
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
  CODER_PORT=${CODER_PORT:-30002}
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
      PY_ARGS+=(
          --model "${MODEL_PATH}"
          --coder-model "${CODER_MODEL_PATH}"
          --start-servers
      )
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
      echo " 手动模式：请确保两个 SGLang 服务器已在运行："
      echo "   Planner 服务器 : port ${PLANNER_PORT}"
      echo "   Coder 服务器   : port ${CODER_PORT}"
      echo ""
      echo " 快速启动示例："
      echo "   python -m sglang.launch_server \\"
      echo "     --model-path ${MODEL_PATH} --port ${PLANNER_PORT} \\"
      echo "     --tp ${TP} --mem-fraction-static ${MEM_FRACTION} \\"
      echo "     --context-length ${CTX_LEN} --trust-remote-code &"
      echo ""
      echo "   python -m sglang.launch_server \\"
      echo "     --model-path ${CODER_MODEL_PATH} --port ${CODER_PORT} \\"
      echo "     --tp ${TP} --mem-fraction-static ${MEM_FRACTION} \\"
      echo "     --context-length ${CTX_LEN} --trust-remote-code &"
      echo "============================================================"
      echo ""
  fi

  # ── Run ───────────────────────────────────────────────────────────────────────

  echo "▶ 开始 32G top-${SAMPLES_PER_PROMPT} 评估..."
  echo "  Planner模型: ${MODEL_PATH}"
  echo "  Coder模型  : ${CODER_MODEL_PATH}"
  echo "  Tokenizer  : ${TOKENIZER_PATH}"
  echo "  输出文件   : ${OUTPUT}"
  echo "  TP         : ${TP}"
  echo "  静态显存占比: ${MEM_FRACTION}"
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
  echo "✓ 32G top-${SAMPLES_PER_PROMPT} 评估完成，结果保存至：${OUTPUT}"