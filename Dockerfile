FROM vllm/vllm-openai:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential pkg-config uuid-dev iptables \
    libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    libnl-3-dev libnet1-dev libcap-dev libaio-dev libbsd-dev libgnutls28-dev \
    python3 python3-protobuf python3-yaml asciidoc xmlto iproute2 \
    && rm -rf /var/lib/apt/lists/*

# criu make install doesn't respect apt asciidoc
RUN pip install asciidoc

ARG CRIU_REPO=https://github.com/checkpoint-restore/criu
ARG CRIU_REF=criu-dev

RUN git clone --branch ${CRIU_REF} --depth 1 --single-branch \
        ${CRIU_REPO} /tmp/criu-src && \
    cd /tmp/criu-src && \
    make && \
    make install-criu install-man && \
    make install-cuda_plugin PLUGINDIR=/usr/local/lib/criu && \
    cd / && rm -rf /tmp/criu-src

RUN git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint /tmp/cuda-checkpoint && \
    cp /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/ && \
    chmod +x /usr/local/bin/cuda-checkpoint && \
    rm -rf /tmp/cuda-checkpoint

RUN git config --global --add safe.directory /opt/vllm
RUN pip uninstall -y vllm
RUN git clone --branch exp/fast-weights https://github.com/j9smith/vllm /opt/vllm
WORKDIR /opt/vllm

RUN pip install setuptools-rust && \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=a970fb5a1a5800c552c74cf3278d6ee7c1c3fca1 \
    pip install -e . --no-build-isolation
RUN pip uninstall -y flashinfer-jit-cache

ENTRYPOINT [ "vllm", "serve", "meta-llama/Llama-3.2-3B-Instruct", "--gpu-memory-utilization", "0.80", "--max-model-len", "8192" ]