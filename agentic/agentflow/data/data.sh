#!/bin/bash

# 1. 创建数据盘上的目标目录（如果不存在）
mkdir -p /root/autodl-tmp/data

# 2. 设置 ModelScope 缓存目录为数据盘路径
export MODELSCOPE_CACHE='/root/autodl-tmp/data'

# 3. 开启 AutoDL 学术加速（下载速度会快很多）
source /etc/network_turbo

# 4. 执行下载命令
modelscope download --model Qwen/Qwen2.5-3B-Instruct
modelscope download --model Qwen/Qwen3.5-4B