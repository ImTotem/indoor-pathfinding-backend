# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mobile-Desktop AR/SLAM Indoor Navigation System - a full-stack application combining:
- **Flutter mobile frontend** for AR-based indoor scanning
- **Python FastAPI backend** for SLAM processing and map management
- **ORB-SLAM3 integration** for visual SLAM localization

## Common Commands

### Backend
```bash
# Install dependencies
pip install -r be/requirements.txt

# Run development server (from project root)
cd be && python main.py
# Or: uvicorn be.main:app --host 0.0.0.0 --port 8000 --reload

# Docker ORB-SLAM3 setup (optional, 30-60 min build)
cd be/slam_engines/orbslam3/docker && ./build.sh && ./run.sh
```

### Frontend
```bash
cd indoor_navigation_app
flutter pub get
flutter run
```

## Architecture

### Backend (`/be`)
```
FastAPI (main.py)
├── Routes: /api/scan/*, /api/localize, /api/path/*, /api/viewer/*
├── SLAM Interface (Factory Pattern)
│   ├── SLAMEngineBase (Abstract)
│   ├── DummySLAMEngine (Testing)
│   └── ORBSlam3Engine (Real SLAM)
├── Storage Manager (file-based, /data/sessions/ and /data/maps/)
└── Services (async SLAM processing)
```

Key files:
- `config/settings.py` - Global settings with .env support
- `slam_interface/factory.py` - SLAM engine factory
- `services/slam_service.py` - Async SLAM processing logic
- `storage/manager.py` - Session/map persistence

### Frontend (`/indoor_navigation_app`)
```
Flutter App
├── Screens: Home, Scanning, Processing, Navigation, MapViewer
├── Services: ApiService, ARService, SensorService, StreamingUploader
├── Models: FrameData, SessionStatus, CameraIntrinsics
└── Config: app_config.dart (API endpoints), constants.dart (Korean UI strings)
```

Key files:
- `lib/services/api_service.dart` - Backend communication
- `lib/services/streaming_uploader.dart` - Chunked frame upload
- `lib/screens/scanning_screen.dart` - AR scanning with real-time capture

### Data Flow
1. Mobile app captures AR frames (image + camera pose + IMU)
2. Streams frames in chunks to backend
3. Backend stores in `/data/sessions/{session_id}/`
4. After scan completion, triggers async SLAM processing
5. SLAM engine generates map (stored in `/data/maps/`)
6. 3D map viewer available via Three.js web interface

## Configuration

- Backend env: `be/.env` (copy from `be/.env.example`)
- SLAM engine selection: Set `SLAM_ENGINE` in .env ("dummy" or "orbslam3")
- ORBSLAM3_PATH: `"docker://orbslam3"` for Docker mode, or local path like `"/path/to/ORB_SLAM3"`
- Mobile API endpoint: `indoor_navigation_app/lib/config/app_config.dart`
- Backend runs on `0.0.0.0:8000`

## ORB-SLAM3 Docker Integration

Docker 모드에서 ORB-SLAM3 실행 시 볼륨 마운트 구조:
- `be/data` → `/data` (컨테이너 내부)
- 호스트 경로 → 컨테이너 경로 변환: `engine.py`의 `_to_container_path()` 메서드

```
Host: /Users/.../be/data/sessions/xxx/images
      ↓ (docker-compose volumes)
Container: /data/sessions/xxx/images
```

### mono_tum 입력 형식
- 3개 인자: `vocabulary`, `settings.yaml`, `sequence_folder`
- `sequence_folder/rgb.txt` 필요 (처음 3줄 주석, 이후 `timestamp images/filename.jpg`)
- Viewer 비활성화 버전으로 빌드됨 (headless 실행)

Key files:
- `slam_engines/orbslam3/engine.py` - Docker/Local 모드 분기, 경로 변환
- `slam_engines/orbslam3/trajectory_parser.py` - rgb.txt 생성, trajectory 파싱
- `slam_engines/orbslam3/templates/camera.yaml.jinja2` - ORB-SLAM3 v1.0 설정 템플릿
- `slam_engines/orbslam3/docker/docker-compose.yml` - 볼륨 마운트 설정

## Current State

- SLAM engine is set to "dummy" for development
- Path calculation endpoint is placeholder implementation
- No authentication (CORS allows all origins)
- File-based storage only (no database)

## Flutter App Scan Quality (ORB-SLAM3 최적화)

스캔 품질 설정 (`lib/config/app_config.dart`):
- `frameIntervalMs: 200` - 5 FPS (품질 우선)
- `imageQuality: 90` - JPEG 품질
- `minMovementDistance: 0.05` - 최소 이동 거리 5cm
- `minRotationAngle: 5.0` - 최소 회전 각도 5°
- `blurThreshold: 100.0` - 블러 감지 임계값

프레임 품질 검사 (`lib/utils/frame_quality_checker.dart`):
- Laplacian variance로 블러 감지
- 움직임 거리/회전 각도 계산
- 이동 속도 체크 (너무 빠르면 거부)
- 키프레임 간 최소 간격 유지

사용자 가이드:
- 움직임 상태 인디케이터 (초록=good, 주황=slow, 빨강=fast/blur)
- 크로스헤어 색상으로 실시간 피드백
- 품질 통과/거부 프레임 수 표시
- 최소 30개 품질 프레임 필요
