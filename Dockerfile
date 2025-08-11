# 使用官方的DeepStream基础镜像
# 该镜像已包含GStreamer, Python绑定(gi)和NVIDIA驱动相关库
FROM nvcr.io/nvidia/deepstream:6.3-gc-triton-devel

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY requirements.txt .
COPY main.py .

# 安装Python依赖
# 基础镜像自带了 numpy，这里安装opencv用于图像处理
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置容器启动时执行的命令
# 将运行main.py脚本
CMD ["python3", "main.py"]
