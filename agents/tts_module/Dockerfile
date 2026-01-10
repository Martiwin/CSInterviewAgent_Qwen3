# 使用轻量级的 Python 镜像
FROM docker.m.daocloud.io/library/python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖（如果 edge-tts 需要的话，这里 slim 版通常够用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖声明并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制脚本
COPY tts_server.py .

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "tts_server.py"]