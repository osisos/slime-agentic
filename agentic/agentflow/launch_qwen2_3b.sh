#!/bin/bash
# AgentFlow Qwen2.5-3B 一键启动脚本
# 自动清理旧进程并启动 agent_flow_qwen25_3b_rl.sh 进行强化学习。
#
# 说明：
#   agent_flow_qwen25_3b_rl.sh 内部会启动外部 Qwen coder/rewarder/verifier
#   SGLang 服务（默认 port=30002），因此本脚本不再重复启动 30002 服务。

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# 提高文件描述符上限，避免 "Too many open files"
ulimit -n 65536 2>/dev/null || true

LOG_DIR=${LOG_DIR:-"/tmp/agentflow_qwen2_3b_logs"}
mkdir -p "$LOG_DIR"

# 默认使用 2 张 GPU：GPU 0/1 训练，其中 agent_flow_qwen25_3b_rl.sh
# 默认会把 GPU 1 作为外部 coder 服务所在 GPU。
TRAIN_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
CODER_PORT=${CODER_PORT:-30002}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_port() {
    local name=$1 host=$2 port=$3 timeout=${4:-600} interval=5 elapsed=0
    log "等待 $name (${host}:${port}) 就绪..."
    while [ $elapsed -lt $timeout ]; do
        if bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
            log "$name 已就绪 (${elapsed}s)"
            return 0
        fi
        if [ -n "${TRAIN_PID:-}" ] && ! kill -0 "$TRAIN_PID" >/dev/null 2>&1; then
            log "ERROR: 训练进程已退出，无法继续等待 $name，请检查日志: $LOG_DIR/train.log"
            return 1
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        if (( elapsed % 30 == 0 )); then
            log "  还在等待 $name... (${elapsed}s)"
        fi
    done
    log "ERROR: $name 在 ${timeout}s 内未能启动，请检查日志: $LOG_DIR"
    return 1
}

# Step 1: 同步清理旧进程，避免端口/GPU 资源冲突
log "清理旧进程..."
pkill -9 sglang 2>/dev/null || true
sleep 1
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true
sleep 2
pkill -9 ray 2>/dev/null || true
pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true

log "等待旧 ray dashboard 端口关闭..."
for _ in $(seq 1 30); do
    if ! bash -c "echo >/dev/tcp/127.0.0.1/8265" 2>/dev/null; then
        break
    fi
    sleep 1
done
log "旧进程已清理。"

# Step 2: 启动 3B 强化学习脚本（跳过训练脚本中的 kill，由本脚本统一清理）
log "启动 Qwen2.5-3B RL 脚本..."
cd "$SLIME_ROOT"
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
CODER_PORT="${CODER_PORT}" \
SKIP_PROCESS_KILL=1 \
    bash "${SCRIPT_DIR}/agent_flow_qwen25_3b_rl.sh" \
    > "$LOG_DIR/train.log" 2>&1 &
TRAIN_PID=$!
log "训练进程 PID=$TRAIN_PID，日志: $LOG_DIR/train.log"
log "GPU: CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}; coder port=${CODER_PORT}"

# Step 3: 等待关键服务就绪
log "等待内部服务启动..."
wait_port "Coder-SGLang-${CODER_PORT}" 127.0.0.1 "${CODER_PORT}" 600
wait_port "ray-dashboard" 127.0.0.1 8265 180
log "服务已就绪，训练正在进行中。"
log "训练日志: tail -f $LOG_DIR/train.log"

# Step 4: 等待训练进程结束（不让 set -e 因训练退出码非零而提前退出）
TRAIN_EXIT=0
wait $TRAIN_PID || TRAIN_EXIT=$?
if [ $TRAIN_EXIT -eq 0 ]; then
    log "训练完成。"
else
    log "训练退出，exit code=$TRAIN_EXIT，请查看日志: $LOG_DIR/train.log"
fi
