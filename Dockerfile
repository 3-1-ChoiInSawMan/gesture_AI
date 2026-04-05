FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

# CUDA 12.1 베이스면 cu121로 맞추는 쪽이 덜 헷갈림
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY pyproject.toml ./
COPY README.md ./

RUN python -m pip install \
    "fastapi>=0.135.1" \
    "faster-whisper>=1.2.1" \
    "openai>=2.30.0" \
    "openai-whisper>=20250625" \
    "pymongo>=4.16.0" \
    "uvicorn[standard]>=0.42.0"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]