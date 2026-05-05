# IMDF Viewer 사용 가이드

Sprint 24 결과물(IMDF zip + map_node/edge GeoJSON + walkable layout)을 브라우저에서 시각적으로 검증하는 개발 도구.

## 시작

```bash
# 1. 환경변수 설정
export INDOOR_DEV_VIEWER_ENABLED=true

# 2. 서버 실행
cd server
uv run uvicorn indoor_server.main:app --reload

# 3. 브라우저 접속
open http://localhost:8000/dev/viewer/
```

## 사용 방법

1. **Bearer 토큰** 입력란에 API 토큰 입력 (기본값: `dev-token`)
2. **scan 드롭다운**에서 빌드 완료된 scan 선택 — 자동으로 scan_id 입력란 채움
3. **Load** 버튼 클릭 → IMDF zip + graph GeoJSON 동시 fetch
4. 왼쪽 패널 **레이어 토글**로 원하는 레이어만 표시
5. **마우스 휠** 줌 / **드래그** 팬
6. **노드 위에 마우스 올리면** tooltip으로 node_id / type / 좌표 확인

## API 경로

| 경로 | 설명 |
|---|---|
| `GET /dev/viewer/` | viewer HTML (StaticFiles) |
| `GET /dev/api/scans` | 빌드 완료 scan 목록 (dev 전용, Bearer 인증) |
| `GET /scan/{id}/imdf` | IMDF zip 다운로드 |
| `GET /scan/{id}/graph` | map_node/edge GeoJSON |

## 레이어 색상

| 레이어 | 색상 | 설명 |
|---|---|---|
| Footprint | 회색 fill + 테두리 | walkable 외곽 영역 |
| Unit | 반투명 파랑 | walkway polygon (footprint과 동일, Sprint 24) |
| Graph edges | 초록 선 | 노드 간 연결 |
| Nodes — skeleton/corridor | 파랑 점 | 뼈대 노드 |
| Nodes — junction | 주황 점 | 분기점 |
| Nodes — endpoint | 회색 점 | 끝점 |
| Nodes — poi | 빨간 큰 점 | POI 노드 |
| Amenity (POI) | 빨간 원 + 텍스트 | IMDF amenity 마커 |
| Anchor | 노란 십자 | scan 원점 (0, 0, z0) |

## 한계 및 주의사항

- **anchor**: Sprint 24 기준 단일 (0, 0, z0) placeholder. GPS anchor 미연동.
- **unit**: Sprint 24에서 footprint과 동일 polygon (`category=walkway`). Sprint 25에서 방/복도 분할 예정.
- **CORS**: 같은 origin 필수. 외부 도메인에서 직접 접근 불가.
- **production 비활성**: `INDOOR_DEV_VIEWER_ENABLED` 미설정 시 `/dev/viewer/`, `/dev/api/scans` 모두 404.
- **좌표계**: `local_metric` (미터 단위 ARKit 로컬). GPS 좌표 아님.
