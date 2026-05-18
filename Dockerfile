# https://hub.docker.com/_/python
FROM python:3.12-slim-bookworm

WORKDIR /opt/es2vec

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU 版 PyTorch，减小镜像体积
# https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 包目录为 /opt/es2vec，上级 /opt 加入路径以支持 import es2vec
ENV PYTHONPATH=/opt
ENV PYTHONUNBUFFERED=1

# 默认仅做离线冒烟；完整流程见 docker-compose.yml
CMD ["python", "cli/smoke_demo.py", "--offline"]
