# Learnings - implement-slam-stubs

## 2026-02-10T07:15:00Z - Task Completion

### Pattern: Query-on-Demand Metadata Access
- The codebase uses a consistent pattern for accessing .db file metadata
- **DatabaseParser** is instantiated and used directly in route handlers
- No caching layer in PostgreSQL - metadata is always queried fresh from .db files
- This pattern is used in: `/api/maps`, `/api/viewer/*`, `/api/localize`

**Reference Implementation**: `be/routes/maps.py:54-67`
```python
db_parser = DatabaseParser()
for db_file in sorted(maps_dir.glob("*.db")):
    db_metadata = await db_parser.parse_database(str(db_file))
    keyframe_count = db_metadata.get('num_keyframes', 0)
```

### RTABMapEngine Integration Pattern
- **Factory pattern** used for engine creation: `SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)`
- **Intrinsics extraction** from map database: `engine.extract_intrinsics_from_db(str(db_path))`
- **Resolution scaling** for mismatched intrinsics: `engine.scale_intrinsics(intrinsics, width, height)`
- **Localize method** signature: `await engine.localize(map_id, [image_bytes], intrinsics=intrinsics)`

**Reference Implementation**: `be/routes/localize.py:54-93`

### Error Handling Strategy
Both endpoints follow consistent error handling:
- **400**: Invalid input (base64 decode error, invalid image data)
- **404**: Resource not found (map doesn't exist)
- **500**: Server errors (intrinsics extraction failure)
- **503**: Service unavailable (timeout, localization failure)

### Architectural Constraints (User Requirements)
1. ✅ **DB 수정 금지**: No PostgreSQL schema changes allowed
2. ✅ **레거시 보존**: Existing `/api/localize` not modified
3. ✅ **중복 허용**: Two localize endpoints can coexist (`/api/localize` and `/api/slam/localize`)
4. ✅ **최소 변경**: Single file modification (`slam_routes.py`)

### Code Quality
- All imports added at top of file (base64, io, PIL.Image, SLAMEngineFactory, DatabaseParser)
- Comprehensive logging for debugging (INFO, WARNING, ERROR levels)
- Graceful fallback for missing/corrupt databases (num_keyframes defaults to 0)
- Proper async/await usage for database queries

### Commits
- `d4855c5`: feat(slam): implement POST /api/slam/localize endpoint
- `379c031`: feat(slam): implement num_keyframes extraction in metadata endpoint

Both commits are atomic and follow conventional commit format.
