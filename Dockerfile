# Gridlock 2.0 — combined API + inference image (backend/ + inference/ + ml/).
# Deploy target: AWS EC2 g4dn.xlarge (T4, 16GB VRAM), Ubuntu 24.04 host w/ NVIDIA driver + nvidia-container-toolkit.
#
# Two Python environments inside one image (same split as local dev):
#   1. System python (numpy>=2): torch/rfdetr/transformers/ultralytics/cv2 + fastapi/uvicorn/supabase
#      — runs the FastAPI app + the main pipeline (everything except SAM-3).
#   2. /app/.venv-sam3 (numpy<2): the official facebookresearch/sam3 package, isolated because
#      it hard-pins numpy<2, incompatible with #1. Talked to over a subprocess worker
#      (ml/src/modules/sam3_worker.py) — see ml/TODO.md Phase 8 for why.

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git curl \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# ---- 1) Main environment: torch+cuda, then the rest of the ML + web stack ----
COPY ml/requirements.txt ./ml-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
# Ubuntu 24.04's system python is "externally managed" (PEP 668) — this container is
# single-purpose, so overriding that guard is the right call (not a shared host install).
# NOTE: don't `pip install --upgrade pip` here — apt's pip 24.0 has no RECORD file, so pip
# can't uninstall itself in place; the apt-shipped pip is fine for everything below.
RUN python -m pip install --break-system-packages torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    python -m pip install --break-system-packages -r ml-requirements.txt -r backend-requirements.txt

# ---- app code ----
COPY ml ./ml
COPY backend ./backend
COPY inference ./inference
COPY vendor ./vendor

# ---- 2) Isolated SAM-3 environment (numpy<2; official package; real triton on Linux) ----
# --system-site-packages reuses the already-installed system torch+cuda (saves ~2.5GB download
# + image size) — safe because torch's numpy interop goes through the buffer protocol, not a
# compiled numpy-version-specific ABI. Only numpy itself + sam3's small extra deps install fresh.
RUN python -m venv --system-site-packages /app/.venv-sam3 && \
    /app/.venv-sam3/bin/pip install --upgrade pip && \
    /app/.venv-sam3/bin/pip install -e "./vendor/sam3[notebooks]" triton

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
