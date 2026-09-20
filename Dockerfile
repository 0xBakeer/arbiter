# Not built by CI and not published anywhere; it is here so the dependency set is written down
# somewhere executable. `./run.sh setup` on the host is the supported path.
#
#   docker build -t arbiter .
#   docker run --gpus all -p 8010:8010 -v "$PWD/models:/app/models" arbiter
#
# The image does not carry the checkpoints. Mount them, or let the container download them on
# first start with HF_TOKEN set.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# python3-dev is not optional: torch routes a few eager operations through Triton, and Triton
# JIT-compiles a C extension against Python.h the first time one of them runs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.12 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN pip install --upgrade pip wheel \
    && pip install torch --index-url https://download.pytorch.org/whl/cu130 \
    && pip install "transformers>=5" safetensors huggingface_hub numpy \
                   fastapi "uvicorn[standard]" "laya==0.3.4" "mcp>=2" pytest httpx

COPY . /app

ENV PORT=8010 \
    HOST=0.0.0.0 \
    DEVICE=cuda \
    ARBITER_MODE=eager \
    ARBITER_DTYPE=autocast \
    ARBITER_MODELS=english,multilingual,typed-decisions \
    ARBITER_MODELS_DIR=/app/models/laya

EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -fsS "http://127.0.0.1:${PORT}/readyz" || exit 1

CMD ["./run.sh", "serve-foreground"]
