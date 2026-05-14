import psutil
import os

# 获取当前进程可用的物理内存
mem = psutil.virtual_memory()
print(f"总内存: {mem.total / (1024**3):.2f} GB")
print(f"当前可用: {mem.available / (1024**3):.2f} GB")

# 如果是容器环境，查看 cgroup 限制
if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
        limit = int(f.read())
        print(f"容器内存限制: {limit / (1024**3):.2f} GB")