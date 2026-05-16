import os
import requests

def download_file(url, target_path):
    target_path = os.path.abspath(os.path.expanduser(target_path))
    print(f"正在下载: {url}")
    print(f"保存至: {target_path}")
    try:
        # 使用流式下载以应对可能的大文件
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # 确保父目录存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunk
                if chunk:
                    f.write(chunk)
        print(f"✅ 下载成功！文件大小: {os.path.getsize(target_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ 下载失败: {e}")

# 1. 基础配置
# 我们使用 zhuzilin 的仓库，因为它们的文件名和结构最稳定
BASE_MIRROR = "https://hf-mirror.com/datasets"

tasks = [
    {
        "url": f"{BASE_MIRROR}/zhuzilin/aime-2024/resolve/main/aime-2024.jsonl",
        "dest": "~/data/aime-2024/aime-2024.jsonl"
    },
    {
        "url": f"{BASE_MIRROR}/zhuzilin/dapo-math-17k/resolve/main/dapo-math-17k.jsonl",
        "dest": "~/data/dapo-math-17k/dapo-math-17k.jsonl"
    }
]

# 2. 执行下载
for task in tasks:
    download_file(task["url"], task["dest"])
    print("-" * 30)

print("\n🎉 所有任务处理完毕！")
