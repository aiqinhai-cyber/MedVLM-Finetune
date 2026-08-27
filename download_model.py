
# 如果申请访问官网上的大模型，非常耗时，所以，这里先下载到本地
import os

# 1. 强制使用国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 彻底关闭导致 Windows 下网络断连的底层加速器 (关键！)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# 3. 清理可能存在的系统代理干扰 
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

from huggingface_hub import snapshot_download

print("开始稳定下载 Qwen2.5-VL-3B 模型到本地...")

snapshot_download(
    repo_id="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    local_dir="D:/models/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=2  # 每次只下载 2 个文件，防止被镜像站风控限流拦截
)
print("模型全量文件下载完毕！可以去启动训练了！")