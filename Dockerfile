# Sprint 65: rtabmap-reprocess CLI 가 필요. base image 를 introlab3it/rtabmap 으로 두고
# 위에 python 3.11 + uv 를 설치한다. 이미지 크지만 의존성 누락 위험 0.
#
# Sprint 67: PyAV (av>=12.0.0) 추가됨. PyAV manylinux 휠이 libav* 정적 번들 → system
# ffmpeg 추가 설치 불필요. uv sync 시 자동으로 휠 설치.
#
# Dev 환경(M4 macOS) 에서는 native homebrew rtabmap + brew ffmpeg 사용 가능 —
# docker 없이도 worker 동작. 본 Dockerfile 은 production / 시연 배포용.

FROM introlab3it/rtabmap:noble

# Python 3.11 + venv 설치
# Ubuntu 24.04 (noble) 기본 저장소는 python3.12만 제공 → deadsnakes PPA 에서 3.11 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python3 → python3.11 우선
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV PYTHONPATH=/app/src:/app/be:/app

# rtabmap binary 가용성 검증 (이미지 빌드 시점)
RUN rtabmap-reprocess 2>&1 | head -1 || true
RUN command -v rtabmap-reprocess

# 의존성 레이어 캐시 (프로젝트 본체 제외 — src/ 복사 전이라 빌드 불가)
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-dev --frozen --no-install-project

# 소스 복사
COPY src ./src
COPY be ./be
COPY alembic ./alembic
COPY alembic.ini .
COPY scripts ./scripts

# 프로젝트 본체 설치
RUN uv sync --no-dev --frozen

# Sprint 87: hloc (SuperPoint + LightGlue + COLMAP wrapper) + pycolmap
# - hloc: PyPI 미배포, github master 만 공식. ETH CV 그룹 표준.
# - pycolmap: COLMAP Python binding (BA / triangulation / DB).
# venv 외부 system pip 가 아닌 venv 내부 pip 로 설치 (uv 와 충돌 회피).
RUN /app/.venv/bin/python -m ensurepip --upgrade 2>&1 | tail -1 \
    && /app/.venv/bin/python -m pip install --no-cache-dir \
        pycolmap \
        "git+https://github.com/cvg/Hierarchical-Localization.git@master"

# venv 의 python/uvicorn/alembic 등을 시스템 PATH 로 노출
# (compose 의 `python -m indoor_server.worker` 가 venv 의 python 으로 resolve 됨)
ENV PATH="/app/.venv/bin:${PATH}"

# 저장소/임시 디렉터리
RUN mkdir -p var/storage var/tmp/uploads be/data/maps be/data/tmp

EXPOSE 8000
