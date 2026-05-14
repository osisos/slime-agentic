#!/bin/bash
# ---------------------------------------------------------
# RTX 5090 一键全自动后台评估脚本
# ---------------------------------------------------------

export NCCL_IGNORE_CPU_AFFINITY=1
export TORCH_CUDA_ARCH_LIST="9.0"

# 1. 暴力清理历史残留进程，防止端口被占用
echo "🧹 清理旧进程..."
fuser -k 30000/tcp 2>/dev/null || true
fuser -k 30002/tcp 2>/dev/null || true
sleep 2

# 2. 路径配置
MODEL_PATH="/root/autodl-tmp/data/models/Qwen/Qwen2.5-3B-Instruct"
MODEL_CODER="/root/autodl-tmp/data/models/Qwen/Qwen3.5-4B"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"

# 3. 在后台独立启动 Planner 模型 (规避 5090 报错的参数)
echo "🚀 正在后台拉起 Planner 模型 (端口 30000)..."
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port 30000 \
  --attention-backend triton \
  --disable-cuda-graph \
  --mem-fraction-static 0.3 \
  --trust-remote-code > planner.log 2>&1 &

# 4. 在后台独立启动 Coder 模型 (规避 5090 报错的参数)
echo "🚀 正在后台拉起 Coder 模型 (端口 30002)..."
python3 -m sglang.launch_server \
  --model-path "${MODEL_CODER}" \
  --port 30002 \
  --attention-backend triton \
  --disable-cuda-graph \
  --mem-fraction-static 0.1 \
  --trust-remote-code > coder.log 2>&1 &

# 5. 智能等待：检测端口是否存活
echo "⏳ 等待模型加载到显存... (大约需要1到3分钟，请勿退出)"
while ! (echo > /dev/tcp/127.0.0.1/30000) >/dev/null 2>&1; do sleep 3; done
echo "✅ Planner (30000) 就绪！"
while ! (echo > /dev/tcp/127.0.0.1/30002) >/dev/null 2>&1; do sleep 3; done
echo "✅ Coder (30002) 就绪！"

# 6. 开始评估
echo "🎯 服务全部上线，开始执行评估..."
python3 "${SCRIPT_DIR}/eval_agentflow.py" \
    --tokenizer  "${MODEL_PATH}" \
    --eval-data  aime /data/aime-2024/aime-2024.jsonl \
    --input-key  prompt \
    --label-key  label \
    --output     "$(dirname "$0")/eval_24G_results.json" \
    --concurrency 16 \
    --max-steps  5 \
    --temperature 0.7 \
    --top-p       0.95 \
    --max-new-tokens 4096 \
    --samples-per-prompt 8 \
    --tp          1 \
    --planner-port 30000 \
    --coder-port   30002 \
    --planner-url  "http://127.0.0.1:30000/generate" \
    --coder-url    "http://127.0.0.1:30002/generate"

# 7. 评估完成后自动杀掉后台模型，释放显存
echo "🛑 评估结束，正在关闭后台模型服务..."
fuser -k 30000/tcp 2>/dev/null || true
fuser -k 30002/tcp 2>/dev/null || true
echo "🎉 完美收工！"