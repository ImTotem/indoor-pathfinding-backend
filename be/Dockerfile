fastapi
uvicorn[standard]
aiofiles
pydantic
python-multipart
pillow
jinja2
python-dotenv
numpy
opencv-contrib-python
asyncpg
--extra-index-url https://download.pytorch.org/whl/cu126
torch
torchvision
ultralytics
lightglue @ git+https://github.com/cvg/LightGlue.git
services:
  rtabmap:
    build:
      context: ./be/slam_engines/rtabmap
      dockerfile: Dockerfile
    image: rtabmap-cuda:latest
    container_name: rtabmap
    environment:
      - LD_LIBRARY_PATH=/usr/local/lib
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./be/data:/data
    working_dir: /data
    command: tail -f /dev/null
    restart: unless-stopped
    networks:
      - indoor-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  web:
    build:
      context: ./be
      dockerfile: Dockerfile
    container_name: indoor-pathfinding-web
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "${SERVER_PORT:-5000}:${SERVER_PORT:-5000}"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      # Server Configuration
      - SERVER_PORT=${SERVER_PORT:-5000}
      
      # SLAM Engine Configuration
      - SLAM_ENGINE=rtabmap
      - RTABMAP_PATH=docker://rtabmap
      - LOG_LEVEL=INFO
      
      # PostgreSQL Configuration
      - POSTGRES_HOST=${POSTGRES_HOST:-indoor-pathfinding-db}
      - POSTGRES_PORT=${POSTGRES_PORT:-5432}
      - POSTGRES_DB=${POSTGRES_DB:-indoor_pathfinding}
      - POSTGRES_USER=${POSTGRES_USER:-indoor}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-indoor1234}
      
      # Fixed Map Mode (optional)
      - USE_FIXED_MAP=${USE_FIXED_MAP:-false}
      - FIXED_MAP_ID=${FIXED_MAP_ID:-}
    volumes:
      - ./be/data:/app/data
      - ${UPLOAD_STORAGE_PATH:?UPLOAD_STORAGE_PATH must be set}:/app/storage/uploads:ro
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - rtabmap
    restart: unless-stopped
    networks:
      - indoor-network
    command: uvicorn main:app --host 0.0.0.0 --port ${SERVER_PORT:-5000} --workers 1

networks:
  indoor-network:
    external: true
    name: indoor-pathfinding-backend_indoor-network
# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl git \
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
