#!/bin/bash

# 设置下载缓存目录为 /data/
export MODELSCOPE_CACHE='/data/'

# 执行下载命令
modelscope download --model Qwen/Qwen3.5-4B
modelscope download --model Qwen/Qwen2.5-3B-Instruct
