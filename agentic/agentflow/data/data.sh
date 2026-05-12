#!/bin/bash

# 创建目标目录
mkdir -p /data/aime-2024/

echo "开始下载 AIME 2024 数据集..."
# 下载 AIME 2024 (假设来源为 Hugging Face)
# 注意：请确保已安装 huggingface-cli
huggingface-cli download \
    m-a-p/AIME_2024 \
    --repo-type dataset \
    --local-dir /data/aime-2024/ \
    --local-dir-use-symlinks False

# 检查并重命名/移动文件以符合你的特定路径要求
if [ -f "/data/aime-2024/AIME_2024.jsonl" ]; then
    mv /data/aime-2024/AIME_2024.jsonl /data/aime-2024/aime-2024.jsonl
fi

echo "------------------------------------"
echo "开始下载 dapo-math-17k 数据集..."

# 下载 dapo-math-17k
huggingface-cli download \
    yibing-du/dapo-math-17k \
    --repo-type dataset \
    --local-dir /data/dapo-math-17k \
    --local-dir-use-symlinks False

echo "------------------------------------"
echo "所有任务已完成！"
echo "文件位置："
echo "- /data/aime-2024/aime-2024.jsonl"
echo "- /data/dapo-math-17k/"