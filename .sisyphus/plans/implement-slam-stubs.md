# Implement SLAM API Stubs (slam_routes.py)

## TL;DR

> **Quick Summary**: Implement 2 placeholder endpoints in `/api/slam/*` to make them functional: localize endpoint (501 stub) and num_keyframes metadata (hardcoded 0).
> 
> **Deliverables**:
> - Working `POST /api/slam/localize` endpoint (JSON-based, calls RTABMapEngine)
> - Working `GET /api/slam/maps/{map_id}/metadata` returning actual keyframe count
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - sequential (single file)
> **Critical Path**: Task 1 → Task 2

---

## Context

### Original Request
사용자: "목업이나 구현되어있지 않은 부분 있는지 확인하고 구현해. 아까 localize가 스텁으로 501이라고 했던거 같은데"

### Analysis Summary

**3개 탐색 에이전트 결과:**
1. **엔드포인트 중복 발견**: `/api/localize` (완전 구현) vs `/api/slam/localize` (501 스텁)
2. **메타데이터 손실 확인**: SLAM 파이프라인에서 `result['metadata']` 버려짐, PostgreSQL에 저장 안됨
3. **기존 패턴 분석**: 코드베이스는 query-on-demand 패턴 사용 (DatabaseParser로 .db 직접 쿼리)

**사용자 제약사항:**
1. DB 수정 금지 → PostgreSQL 스키마 변경 불가
2. 레거시 건드리지 마 → 기존 `/api/localize` 수정 금지
3. 중복 OK → `/api/slam/localize` 새로 구현
4. 동작만 하면 됨 → 최소 변경

### Research Findings

**Existing Implementation:**
- `RTABMapEngine.localize()` - 완전 구현됨 (lines 657-846 in engine.py)
- `DatabaseParser.parse_database()` - .db 파일에서 keyframes 추출 가능 (line 79-134 in database_parser.py)
- `/api/localize` - 멀티파트 폼 데이터, 1-5 이미지, 완전 동작 (localize.py)

**Files to Modify:**
- `be/routes/slam_routes.py` (단 1개 파일만 수정)

---

## Work Objectives

### Core Objective
`slam_routes.py`의 2개 플레이스홀더를 기존 SLAM 엔진을 활용하여 실제 동작하도록 구현

### Concrete Deliverables
1. `POST /api/slam/localize` 엔드포인트 - RTABMapEngine 호출하여 실제 pose 반환
2. `GET /api/slam/maps/{map_id}/metadata` - DatabaseParser로 실제 keyframe 개수 반환

### Definition of Done
- [x] `curl -X POST /api/slam/localize` → 200 응답, pose 데이터 반환 (501 아님)
- [x] `curl /api/slam/maps/{map_id}/metadata` → `num_keyframes > 0` (실제 값)
- [x] 레거시 엔드포인트 수정 없음 (`/api/localize` 그대로)

### Must Have
- `/api/slam/localize`: JSON body로 map_id + base64 이미지 받기
- `/api/slam/localize`: RTABMapEngine.localize() 호출
- `num_keyframes`: DatabaseParser.parse_database() 호출하여 실제 값 반환

### Must NOT Have (Guardrails)
- ❌ PostgreSQL 스키마 변경 (db 수정 금지)
- ❌ 레거시 파일 수정 (localize.py, scan.py, 등)
- ❌ job_queue.py 수정 (메타데이터 저장 로직 추가하지 말 것)
- ❌ 새 의존성 추가 (requirements.txt 수정 금지)

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: NO
- **Framework**: None

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

> QA scenarios는 PRIMARY 검증 방법입니다.
> 에이전트가 curl/Playwright로 직접 API를 호출하고 응답을 검증합니다.

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **API/Backend** | Bash (curl) | Send requests, parse responses, assert fields |

**Each Scenario MUST Follow This Format:**

```
Scenario: [Descriptive name]
  Tool: Bash (curl)
  Preconditions: [What must be true before running]
  Steps:
    1. [Exact curl command with specific data]
    2. [Assertion with exact expected value]
  Expected Result: [Concrete, observable outcome]
  Failure Indicators: [What would indicate failure]
  Evidence: [Response body capture path]
```

---

## Execution Strategy

### Parallel Execution Waves

> Sequential execution (single file modification)

```
Wave 1 (Start Immediately):
└── Task 1: Implement /api/slam/localize endpoint

Wave 2 (After Wave 1):
└── Task 2: Implement num_keyframes extraction

Critical Path: Task 1 → Task 2
Parallel Speedup: N/A (sequential)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | None (same file) |
| 2 | 1 | None | None (same file) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | task(category="quick", load_skills=[], run_in_background=false) |
| 2 | 2 | task(category="quick", load_skills=[], run_in_background=false) |

---

## TODOs

> Implementation only (no tests). ONE task per endpoint.

- [x] 1. Implement POST /api/slam/localize endpoint

  **What to do**:
  - Remove 501 HTTPException from line 263-266 in slam_routes.py
  - Import required modules: `SLAMEngineFactory`, `settings`, `base64`, `io`, `PIL.Image`
  - Parse `SLAMLocalizeRequest.image` (base64 string) → bytes
  - Call `SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)`
  - Extract intrinsics from map database: `engine.extract_intrinsics_from_db()`
  - Scale intrinsics if needed: `engine.scale_intrinsics()`
  - Call `engine.localize(map_id, [image_bytes], intrinsics=intrinsics)`
  - Return `SLAMLocalizeResponse` with pose + confidence

  **Must NOT do**:
  - Modify `/api/localize` in localize.py (레거시 건드리지 말 것)
  - Add new database columns
  - Change request/response models (SLAMLocalizeRequest/Response 그대로 사용)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single endpoint implementation, straightforward logic
  - **Skills**: []
    - Reason: No special skills needed (standard FastAPI + existing engine)
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (sequential)
  - **Blocks**: Task 2
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `be/routes/localize.py:54-93` - RTABMapEngine usage pattern (create engine, extract intrinsics, scale intrinsics, call localize)
  - `be/routes/localize.py:36-45` - Image bytes parsing from uploaded files
  - `be/routes/localize.py:68-90` - Intrinsics scaling pattern with resolution mismatch handling

  **API/Type References** (contracts to implement against):
  - `be/models/slam_api.py:SLAMLocalizeRequest` - Input schema (map_id: str, image: str base64, camera_intrinsics: optional dict)
  - `be/models/slam_api.py:SLAMLocalizeResponse` - Output schema (pose: dict, confidence: float)

  **Engine References**:
  - `be/slam_engines/rtabmap/engine.py:657-846` - RTABMapEngine.localize() method signature and return value
  - `be/slam_engines/rtabmap/engine.py:82-215` - extract_intrinsics_from_db() method
  - `be/slam_engines/rtabmap/engine.py:217-295` - scale_intrinsics() method

  **Factory Pattern**:
  - `be/slam_interface/factory.py` - SLAMEngineFactory.create(engine_type)
  - `be/config/settings.py` - settings.SLAM_ENGINE_TYPE, settings.MAPS_DIR

  **WHY Each Reference Matters**:
  - `localize.py:54-93` - Shows exact pattern for calling engine.localize() with intrinsics extraction/scaling
  - `slam_api.py` - Defines request/response models you must use (don't create new ones)
  - `engine.py:657` - Shows localize() expects List[bytes], returns dict with pose/confidence
  - Base64 decoding needed since SLAMLocalizeRequest.image is string, not bytes

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Successful localization returns pose and confidence
    Tool: Bash (curl)
    Preconditions: 
      - Backend running on localhost:8000
      - Map exists: /data/maps/test_map.db (from previous SLAM processing)
      - Test image available: /data/test_images/query.jpg
    Steps:
      1. Encode test image to base64:
         BASE64_IMG=$(base64 -i /data/test_images/query.jpg)
      2. Send POST request:
         curl -s -X POST http://localhost:8000/api/slam/localize \
           -H "Content-Type: application/json" \
           -d "{\"map_id\":\"test_map\",\"image\":\"$BASE64_IMG\",\"camera_intrinsics\":{\"fx\":500,\"fy\":500,\"cx\":320,\"cy\":240}}" \
           -w "\n%{http_code}" -o /tmp/localize_response.json
      3. Assert: HTTP status is 200 (not 501)
      4. Assert: Response contains "pose" key
      5. Assert: Response.pose has keys: x, y, z, qx, qy, qz, qw
      6. Assert: Response contains "confidence" key (float 0.0-1.0)
      7. Parse JSON: jq '.pose.x' /tmp/localize_response.json → valid number
    Expected Result: 200 OK, pose data with 7 floats + confidence
    Failure Indicators: 501 status, missing pose keys, confidence out of range
    Evidence: /tmp/localize_response.json

  Scenario: Localization fails with non-existent map
    Tool: Bash (curl)
    Preconditions: Backend running, map_id "nonexistent" does not exist
    Steps:
      1. curl -s -X POST http://localhost:8000/api/slam/localize \
           -H "Content-Type: application/json" \
           -d '{"map_id":"nonexistent","image":"base64data"}' \
           -w "\n%{http_code}"
      2. Assert: HTTP status is 404
      3. Assert: Response contains "detail" field mentioning map not found
    Expected Result: 404 Not Found with error message
    Failure Indicators: 500 error, 501 status
    Evidence: Error response body captured

  Scenario: Invalid base64 image returns 400
    Tool: Bash (curl)
    Preconditions: Backend running, valid map exists
    Steps:
      1. curl -s -X POST http://localhost:8000/api/slam/localize \
           -H "Content-Type: application/json" \
           -d '{"map_id":"test_map","image":"invalid_base64!!!"}' \
           -w "\n%{http_code}"
      2. Assert: HTTP status is 400 or 422 (validation error)
      3. Assert: Response mentions base64 or image decoding error
    Expected Result: 400/422 error with meaningful message
    Failure Indicators: 500 error, successful 200 response
    Evidence: Error response body
  ```

  **Evidence to Capture:**
  - [ ] Response bodies in /tmp/localize_*.json
  - [ ] HTTP status codes verified (200, 404, 400)
  - [ ] Pose field validation (7 float values)

  **Commit**: YES
  - Message: `feat(slam): implement POST /api/slam/localize endpoint`
  - Files: `be/routes/slam_routes.py`
  - Pre-commit: None

---

- [x] 2. Implement num_keyframes extraction in metadata endpoint

  **What to do**:
  - Import `DatabaseParser` from `slam_engines.rtabmap.database_parser`
  - Replace line 225 `num_keyframes = 0` with actual extraction logic
  - Calculate db_path: `settings.MAPS_DIR / f"{map_id}.db"`
  - Create `DatabaseParser()` instance
  - Call `await parser.parse_database(str(db_path))`
  - Extract `num_keyframes` from result dict
  - Handle exceptions: FileNotFoundError if .db doesn't exist

  **Must NOT do**:
  - Add metadata columns to PostgreSQL (db 수정 금지)
  - Modify job_queue.py to store metadata
  - Cache metadata in PostgreSQL

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple database query, existing parser available
  - **Skills**: []
    - Reason: Standard async Python, no special domain knowledge
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 1)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 1 (same file modification)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `be/routes/maps.py:54-67` - DatabaseParser usage pattern (create parser, call parse_database, extract num_keyframes)
  - `be/routes/viewer.py:105-109` - Another example of DatabaseParser usage in routes

  **API/Type References**:
  - `be/models/slam_api.py:MapMetadata` - Response model with num_keyframes field
  - `be/routes/slam_routes.py:211-218` - Existing get_job_status call showing current metadata endpoint structure

  **Database Parser References**:
  - `be/slam_engines/rtabmap/database_parser.py:79-134` - parse_database() method (async, returns dict with num_keyframes)
  - `be/slam_engines/rtabmap/database_parser.py:92-93` - SQL query: "SELECT COUNT(*) FROM Node" for keyframe count

  **Configuration References**:
  - `be/config/settings.py` - settings.MAPS_DIR path for .db files

  **WHY Each Reference Matters**:
  - `maps.py:54-67` - Shows exact pattern: create DatabaseParser(), await parse_database(), get num_keyframes
  - `database_parser.py:79` - Shows parse_database() is async and returns dict with 'num_keyframes' key
  - Current slam_routes.py:223-225 - TODO comment explicitly asks for this implementation

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Metadata returns actual keyframe count for existing map
    Tool: Bash (curl)
    Preconditions:
      - Backend running on localhost:8000
      - Map exists: /data/maps/test_map.db with known keyframe count
      - PostgreSQL has job record for test_map
    Steps:
      1. Count actual keyframes in database:
         EXPECTED_COUNT=$(sqlite3 /data/maps/test_map.db "SELECT COUNT(*) FROM Node")
      2. Fetch metadata:
         curl -s http://localhost:8000/api/slam/maps/test_map/metadata \
           -o /tmp/metadata_response.json
      3. Assert: HTTP status is 200
      4. Extract num_keyframes: 
         RETURNED_COUNT=$(jq '.num_keyframes' /tmp/metadata_response.json)
      5. Assert: RETURNED_COUNT equals EXPECTED_COUNT (not 0)
      6. Assert: RETURNED_COUNT > 0 (not placeholder)
      7. Assert: Response contains map_id, session_id, status fields
    Expected Result: num_keyframes matches actual Node count in .db
    Failure Indicators: num_keyframes=0, mismatch with database, 500 error
    Evidence: /tmp/metadata_response.json

  Scenario: Metadata returns 404 for non-existent map
    Tool: Bash (curl)
    Preconditions: Backend running, map_id "fake_map" does not exist
    Steps:
      1. curl -s http://localhost:8000/api/slam/maps/fake_map/metadata \
           -w "\n%{http_code}"
      2. Assert: HTTP status is 404
      3. Assert: Response contains "detail" field mentioning map not found
    Expected Result: 404 Not Found
    Failure Indicators: 500 error, 200 with placeholder data
    Evidence: Error response body

  Scenario: Metadata handles empty database gracefully
    Tool: Bash (curl)
    Preconditions: 
      - Backend running
      - Empty .db file created: touch /data/maps/empty_map.db
      - PostgreSQL has job record for empty_map
    Steps:
      1. curl -s http://localhost:8000/api/slam/maps/empty_map/metadata
      2. Assert: HTTP status is 200 (not 500)
      3. Assert: num_keyframes=0 (valid for empty database)
      4. No crash or internal server error
    Expected Result: 200 OK, num_keyframes=0 for valid empty database
    Failure Indicators: 500 error, crash
    Evidence: Response body
  ```

  **Evidence to Capture:**
  - [ ] SQLite query output (actual keyframe count)
  - [ ] API response with num_keyframes field
  - [ ] Comparison verification (API value == DB value)

  **Commit**: YES
  - Message: `feat(slam): implement num_keyframes extraction in metadata endpoint`
  - Files: `be/routes/slam_routes.py`
  - Pre-commit: None

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(slam): implement POST /api/slam/localize endpoint` | be/routes/slam_routes.py | curl POST → 200 OK |
| 2 | `feat(slam): implement num_keyframes extraction in metadata endpoint` | be/routes/slam_routes.py | curl GET → num_keyframes > 0 |

---

## Success Criteria

### Verification Commands
```bash
# Test localize endpoint (not 501 anymore)
curl -X POST http://localhost:8000/api/slam/localize \
  -H "Content-Type: application/json" \
  -d '{"map_id":"test_map","image":"<base64>"}' | jq .pose

# Expected: {"x": 1.23, "y": 4.56, ...} (not 501 error)

# Test metadata endpoint (not 0 anymore)
curl http://localhost:8000/api/slam/maps/test_map/metadata | jq .num_keyframes

# Expected: 42 (actual count, not hardcoded 0)
```

### Final Checklist
- [x] `/api/slam/localize` returns 200 OK (not 501)
- [x] Localize response has pose with 7 float fields
- [x] `num_keyframes` returns actual database count
- [x] No modifications to `/api/localize` (레거시 보존)
- [x] No PostgreSQL schema changes (db 수정 금지)
