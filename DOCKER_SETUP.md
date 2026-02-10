# Docker Compose 환경 설정

이 프로젝트는 Docker Compose를 사용하여 RTABMap SLAM 엔진과 FastAPI 웹 서버를 실행합니다.

## 네트워크 구조

- **네트워크 이름**: `indoor-pathfinding-backend_indoor-network`
- **외부 네트워크**: 기존 PostgreSQL 컨테이너와 동일한 네트워크 사용
- **서비스 간 통신**: 
  - `web` → `rtabmap` (Docker exec를 통한 SLAM 명령 실행)
  - `web` → `postgres` (PostgreSQL 연결, 외부 네트워크)

## 사전 준비

### 1. 외부 네트워크 생성 (최초 1회)

```bash
docker network create indoor-pathfinding-backend_indoor-network
```

### 2. PostgreSQL 컨테이너 실행 (별도 관리)

PostgreSQL은 별도의 docker-compose로 관리되며, 동일한 네트워크에 연결되어야 합니다:

```yaml
# 예시: postgres docker-compose.yml
services:
  postgres:
    image: postgres:15
    container_name: postgres
    environment:
      POSTGRES_DB: slam_db
      POSTGRES_USER: slam_service
      POSTGRES_PASSWORD: changeme
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - indoor-network

networks:
  indoor-network:
    external: true
    name: indoor-pathfinding-backend_indoor-network
```

### 3. 환경 변수 설정

루트 디렉토리의 `.env` 파일을 확인하고 필요시 수정:

```bash
# .env 파일 확인
cat .env

# PostgreSQL 비밀번호 등 수정 필요시
nano .env
```

## 서비스 구성

### 1. rtabmap

- **이미지**: `introlab3it/rtabmap_ros:noetic-latest`
- **역할**: SLAM 엔진 (매핑 및 로컬라이제이션)
- **볼륨**: `./be/data:/data` (맵 데이터 영구 저장)
- **상태**: 백그라운드 실행 (`tail -f /dev/null`)

### 2. web

- **베이스**: Python 3.10-slim
- **역할**: FastAPI 웹 서버
- **포트**: `8000:8000` (호스트:컨테이너)
- **실행**: `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1`
- **의존성**: rtabmap (SLAM 명령 실행), postgres (데이터베이스)

## 실행 방법

### 전체 스택 시작

```bash
# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 특정 서비스 로그만 확인
docker-compose logs -f web
```

### 개별 서비스 재시작

```bash
# 웹 서버만 재시작 (코드 변경 후)
docker-compose restart web

# RTABMap만 재시작
docker-compose restart rtabmap
```

### 이미지 재빌드

```bash
# 웹 서버 이미지 재빌드
docker-compose build web

# 재빌드 후 실행
docker-compose up -d --build web
```

### 서비스 중지 및 삭제

```bash
# 중지만
docker-compose stop

# 중지 + 컨테이너 삭제
docker-compose down

# 볼륨까지 삭제 (주의: 맵 데이터 삭제됨)
docker-compose down -v
```

## 데이터 볼륨

### 공유 볼륨 구조

```
./be/data/  (호스트)
├── maps/          → rtabmap, web 모두 접근 가능
│   └── <map_id>/
│       └── rtabmap.db
├── sessions/      → 스캔 세션 데이터
└── temp/          → 임시 파일
```

- **rtabmap 컨테이너**: `/data`로 마운트
- **web 컨테이너**: `/app/data`로 마운트
- **동일 디렉토리 공유**: 웹 서버가 생성한 세션을 rtabmap이 읽고, rtabmap이 생성한 맵을 웹 서버가 읽음

## 네트워크 통신

### 서비스 간 연결

1. **web → rtabmap**
   ```python
   # RTABMapEngine에서 docker exec 실행
   docker exec rtabmap rtabmap-... --params ...
   ```

2. **web → postgres**
   ```python
   # main.py에서 asyncpg 연결
   host="postgres"  # 서비스 이름으로 자동 DNS 해석
   ```

### 외부 접근

- **API 서버**: `http://localhost:8000`
- **Health Check**: `http://localhost:8000/`
- **API Docs**: `http://localhost:8000/docs`

## 로그 및 디버깅

### 컨테이너 내부 접속

```bash
# 웹 서버 접속
docker-compose exec web bash

# RTABMap 접속
docker-compose exec rtabmap bash

# Python 인터프리터 실행 (웹 서버)
docker-compose exec web python
```

### 환경 변수 확인

```bash
# 웹 서버 환경 변수
docker-compose exec web env | grep POSTGRES
docker-compose exec web env | grep SLAM
```

### 볼륨 마운트 확인

```bash
# 웹 서버 데이터 디렉토리
docker-compose exec web ls -la /app/data

# RTABMap 데이터 디렉토리
docker-compose exec rtabmap ls -la /data
```

## 트러블슈팅

### 1. 네트워크 연결 오류

```bash
# 네트워크 존재 확인
docker network ls | grep indoor-pathfinding-backend_indoor-network

# 없으면 생성
docker network create indoor-pathfinding-backend_indoor-network
```

### 2. PostgreSQL 연결 실패

```bash
# PostgreSQL 컨테이너 실행 확인
docker ps | grep postgres

# 동일 네트워크 확인
docker inspect postgres | grep indoor-pathfinding-backend_indoor-network

# 웹 서버에서 연결 테스트
docker-compose exec web ping postgres
```

### 3. RTABMap 명령 실패

```bash
# RTABMap 컨테이너 상태 확인
docker-compose exec rtabmap which rtabmap-console

# 볼륨 마운트 확인
docker-compose exec rtabmap ls -la /data/maps
```

### 4. 포트 충돌

```bash
# 8000번 포트 사용 중인 프로세스 확인
lsof -i :8000

# docker-compose.yml에서 다른 포트로 변경
ports:
  - "8001:8000"  # 호스트 포트 변경
```

## 개발 워크플로우

### 1. 코드 수정 후 재시작

```bash
# 방법 1: 재시작 (빠름)
docker-compose restart web

# 방법 2: 재빌드 + 재시작 (requirements.txt 변경 시)
docker-compose up -d --build web
```

### 2. 로그 실시간 모니터링

```bash
# 터미널 1: 로그 모니터링
docker-compose logs -f web

# 터미널 2: 코드 수정 → 재시작
docker-compose restart web
```

### 3. 맵 데이터 백업

```bash
# 맵 데이터 압축
tar -czf slam_maps_backup_$(date +%Y%m%d).tar.gz be/data/maps/

# 복원
tar -xzf slam_maps_backup_20260210.tar.gz
```

## 성능 최적화

### 1. 단일 워커 유지

uvicorn은 **반드시 단일 워커**로 실행:
```yaml
command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

이유: SLAMJobQueue가 싱글톤으로 동작해야 함 (동시 처리 방지)

### 2. 메모리 제한 (선택사항)

```yaml
services:
  web:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 512M
```

### 3. 볼륨 캐싱 (macOS/Windows)

```yaml
volumes:
  - ./be/data:/app/data:cached  # 읽기 우선
```

## 프로덕션 배포

### 1. 환경 변수 보안

```bash
# .env 파일 권한 제한
chmod 600 .env

# 민감 정보는 Docker Secrets 사용 권장
docker secret create postgres_password /path/to/password.txt
```

### 2. 리버스 프록시 (nginx)

```nginx
upstream indoor-pathfinding {
    server localhost:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://indoor-pathfinding;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. 자동 재시작

docker-compose.yml에 이미 설정됨:
```yaml
restart: unless-stopped
```

시스템 재부팅 시 자동으로 컨테이너 재시작됨.
