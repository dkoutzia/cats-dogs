FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV HF_HOME=/app/.cache/huggingface
ENV MODEL_ID=dkoutzia97/cat-dog-vit

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip


RUN python -m pip install --no-cache-dir \
    -r requirements.txt

COPY app ./app

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000
EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]