# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI (needed to exec into rtabmap container)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar xz --strip-components=1 -C /usr/local/bin docker/docker

# uv (10-100x faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (BuildKit cache keeps downloads across builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt && \
    uv pip install --system --reinstall opencv-contrib-python

# Copy application code
COPY . .

RUN mkdir -p /app/data

EXPOSE ${SERVER_PORT:-5000}
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${SERVER_PORT:-5000}", "--workers", "1"]
