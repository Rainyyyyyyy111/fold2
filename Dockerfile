# ==========================================
# 阶段 1：编译环境 (Builder Stage) - 极度臃肿，但用完即丢
# ==========================================
FROM --platform=linux/amd64 nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 curl git build-essential cmake python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && \
    mkdir -p /opt/conda/bin && mv ./bin/micromamba /opt/conda/bin/

WORKDIR /app
COPY Code/FoldFlow /app/Code/FoldFlow

# 1. 剔除 -e .，防止 Singularity 只读挂载报错
# 2. 安装全部依赖，编译 C++ 包
# 3. 静态安装项目，清理多余缓存
RUN cd /app/Code/FoldFlow && \
    sed -i '/- -e ./d' environment.yaml && \
    micromamba create -y -f environment.yaml && \
    micromamba run -n foldflow-env pip install --no-cache-dir . && \
    micromamba clean --all --yes

# 关键：删除预编译的 __pycache__ 和 pip 缓存，节省几百 MB
RUN find /opt/conda/envs/foldflow-env -type d -name "__pycache__" -exec rm -rf {} +

# ==========================================
# 阶段 2：运行环境 (Runtime Stage) - 极限小巧，最终推送到 GitHub 的产物
# ==========================================
# 改用 runtime 镜像，没有巨无霸级的 nvcc 编译器，体积立减数 GB！
FROM --platform=linux/amd64 nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/envs/foldflow-env/bin:$PATH

# 仅安装运行时必须的极少量系统库 (如 libgomp1 对于大矩阵运算必不可少)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段，直接把编译好的整个虚拟环境"端过来"
COPY --from=builder /opt/conda/envs/foldflow-env /opt/conda/envs/foldflow-env
# 拷入我们的项目代码
COPY --from=builder /app /app

WORKDIR /app

# 设置默认启动命令
WORKDIR /app/Code/FoldFlow
CMD ["python", "runner/inference.py", "--help"]