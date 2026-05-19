#!/bin/bash

set -ex

# create conda
yes '' | "${SHELL}" <(curl -sL https://micro.mamba.pm/install.sh)
export PS1=tmp
mkdir -p /root/.cargo/
touch /root/.cargo/env
source ~/.bashrc

# ==================== 网络与源修复模块 ====================
export NO_PROXY="localhost,127.0.0.1,.tsinghua.edu.cn,.sustech.edu.cn,.anaconda.org,.anaconda.com,${NO_PROXY:-}"
export no_proxy=$NO_PROXY

# 彻底清除默认 channel，防止 mamba 偷偷连官方源
cat <<EOF > ~/.condarc
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
show_channel_urls: true
EOF

# pip 默认使用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# ==========================================================

# 1. 创建环境：强制使用清华 conda-forge 源，并覆盖默认 channels
micromamba create -n slime python=3.12 pip -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ --override-channels -y
micromamba clean -a -y
micromamba activate slime

export CUDA_HOME="$CONDA_PREFIX"
export SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR

# 2. 安装 CUDA 12.9：强制使用南科大 nvidia 源，覆盖默认 channels
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c https://mirrors.sustech.edu.cn/anaconda-extra/cloud/nvidia/label/cuda-12.9.1/ --override-channels -y
micromamba clean -a -y

# 3. 安装 cuDNN：强制使用清华源，覆盖默认 channels
micromamba install -n slime cudnn -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ --override-channels -y
micromamba clean -a -y

# ==================== 下面是原有的 pip 安装部分 ====================
pip install --no-cache-dir cuda-python==13.1.0
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# install sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout ${SGLANG_COMMIT}
pip install --no-cache-dir -e "python[all]"

pip install --no-cache-dir cmake ninja

# flash attn
MAX_JOBS=64 pip -v install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

pip install --no-cache-dir git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install --no-cache-dir flash-linear-attention==0.4.0

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall

pip install --no-cache-dir git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install --no-cache-dir nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# megatron
cd $BASE_DIR
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
  cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
  pip install --no-cache-dir -e .

# install slime and apply patches
if [ ! -d "$BASE_DIR/slime" ]; then
  cd $BASE_DIR
  git clone  https://github.com/THUDM/slime.git
  cd slime/
  export SLIME_DIR=$BASE_DIR/slime
  pip install --no-cache-dir -e .
else
  export SLIME_DIR=$BASE_DIR/
  pip install --no-cache-dir -e .
fi

# https://github.com/pytorch/pytorch/issues/168167
pip install --no-cache-dir nvidia-cudnn-cu12==9.16.0.29
pip install --no-cache-dir "numpy<2"

# apply patch
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/v0.5.7/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.7/megatron.patch