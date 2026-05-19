# https://hub.docker.com/_/python
FROM python:3.12-slim-bookworm

WORKDIR /opt/es2vec

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 国内 ECS：勿从 PyPI 主站装 torch（Linux 默认 CUDA 版 500MB+，还会拉数 GB nvidia-*）
# 先装 CPU 版（约 150–200MB），再装其余依赖并锁定 torch 版本，避免被 requirements 升级回 CUDA 版
# PyPI 清华：https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
# PyTorch CPU 阿里云：https://mirrors.aliyun.com/pytorch-wheels/cpu/
# PyTorch CPU 官方（海外）：https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.5.1
ARG PYTORCH_CPU_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn \
    && pip install --no-cache-dir "torch==${TORCH_VERSION}" \
        -f "${PYTORCH_CPU_FIND_LINKS}" \
    && echo "torch==${TORCH_VERSION}" > /tmp/torch-constraint.txt \
    && pip install --no-cache-dir -r requirements.txt -c /tmp/torch-constraint.txt

COPY . .

# 包目录为 /opt/es2vec，上级 /opt 加入路径以支持 import es2vec
ENV PYTHONPATH=/opt
ENV PYTHONUNBUFFERED=1

# 默认仅做离线冒烟；完整流程见 docker-compose.yml
CMD ["python", "cli/smoke_demo.py", "--offline"]
