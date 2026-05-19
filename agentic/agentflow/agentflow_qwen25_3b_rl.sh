#!/bin/bash

# AgentFlow Qwen2.5-3B RL on 2 H100/H800 GPUs.
#
# Environment split:
#   - Slime/Ray/Megatron training runs in the caller's training environment.
#   - External coder/rewarder/verifier inference is started by launch_qwen2_3b.sh.
#
# GPU layout:
#   - GPU 0/1: Megatron actor training with TP=2.
#   - GPU 0  : Slime-managed Qwen2.5-3B rollout/planner engine.
#   - GPU 1  : Reserved for the external Qwen 4B coder/rewarder/verifier SGLang server.

if [ "${SKIP_PROCESS_KILL}" != "1" ]; then
    pkill -9 sglang
    sleep 3
    ray stop --force
    pkill -9 ray
    pkill -9 python
    sleep 3
    pkill -9 ray
    pkill -9 python
fi

set -ex

SAVE_TRAJECTORY=${SAVE_TRAJECTORY:-"0"}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"9T9qsYeuQqoVQeZno7JmW"}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

TRAJECTORIES_DIR="${SCRIPT_DIR}/trajectories"
if [ "${SAVE_TRAJECTORY:-0}" = "1" ]; then
    rm -rf "${TRAJECTORIES_DIR}"
    mkdir -p "${TRAJECTORIES_DIR}"
fi

export PYTHONBUFFERED=16
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
TRAIN_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

MODEL_PATH=${MODEL_PATH:-"/root/blockdata/models/Qwen2.5-3B-Instruct"}
REF_PATH=${REF_PATH:-"/root/blockdata/models/qwen2.5_3b_dist/"} # 改 dist
SAVE_PATH=${SAVE_PATH:-"/root/data/models/AgentFlow_Qwen25-3B-RL/"} # 这个需要改

# Reserve GPU 1 from Slime rollout placement so the external 4B coder can stay resident.
SGLANG_CONFIG=${SGLANG_CONFIG:-"${SCRIPT_DIR}/sglang_qwen25_3b_2gpu_with_coder.yaml"}
cat > "${SGLANG_CONFIG}" <<EOF
sglang:
  - name: default
    model_path: ${MODEL_PATH}
    num_gpus_per_engine: 1
    engine_groups:
      - worker_type: regular
        num_gpus: 1
        num_gpus_per_engine: 1
        overrides:
          mem_fraction_static: 0.58
          context_length: 32768
      - worker_type: placeholder
        num_gpus: 1
EOF

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${REF_PATH}"
   --save "${SAVE_PATH}"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data /root/blockdata/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --reward-key score
   --num-epoch 1
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 0.7
   --global-batch-size 16
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/blockdata/data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 32768
   --eval-top-p 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 12288
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=()

SWANLAB_ARGS=(
   --use-swanlab
   --swanlab-project AgentFlow_pro
   --swanlab-experiment-name AgentFlow_pro-Qwen25-3B-RL
   --swanlab-mode cloud
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-config "${SGLANG_CONFIG}"
   --sglang-mem-fraction-static 0.58
   --sglang-context-length 32768
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path rollout.generate
   --custom-rm-path rollout.reward_func
   --custom-eval-rollout-log-function-path rollout.eval_log
   --custom-convert-samples-to-train-data-path custom_convert.custom_convert
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SAVE_TRAJECTORY\": \"${SAVE_TRAJECTORY}\",
    \"SWANLAB_API_KEY\": \"${SWANLAB_API_KEY}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --num-gpus-per-node 2 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SWANLAB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
