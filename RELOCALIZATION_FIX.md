# RTAB-Map Relocalization Descriptor Mismatch Fix

## Problem

When using the "이미지로 현재 위치 찾기" (localization) feature, RTAB-Map threw errors:

```
[ERROR] VWDictionary.cpp:931::addNewWords() Descriptors (size=64) are not the 
same size as already added words in dictionary (size=32)
```

This prevented successful relocalization against existing maps.

## Root Cause Analysis

### Descriptor Size Mismatch

RTAB-Map uses BRIEF descriptors for feature matching. The descriptor size must match between:
1. **Map creation** (when building the database)
2. **Relocalization** (when localizing against that map)

Our setup had a mismatch:

| Database | BRIEF/Bytes | Created With |
|----------|-------------|--------------|
| `260202-202240.db` | 64 | iOS RTAB-Map app |
| `map_session_20260209_191433_b716707e.db` | 32 | Our Flutter app |

### The Bug

In `engine.py`, the `localize()` method used hardcoded parameters:

```python
# OLD CODE (buggy)
reloc_params = constants.DEFAULT_PARAMS.copy()  # Always uses BRIEF/Bytes=64
```

When relocalizing against our Flutter app's map (32-byte descriptors):
1. RTAB-Map loads the map database (expects 32-byte descriptors)
2. Extracts features from query images using DEFAULT_PARAMS (64 bytes)
3. Tries to add 64-byte descriptors to a dictionary expecting 32 bytes
4. **Error**: Size mismatch detected

## Solution

### Extract Parameters from Target Database

Added `extract_params_from_db()` method to read the target map's configuration:

```python
def extract_params_from_db(self, db_path: str) -> dict:
    """Extract SLAM parameters from database Info table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT parameters FROM Info LIMIT 1")
    row = cursor.fetchone()
    
    param_str = row[0]  # Semicolon-separated key:value pairs
    params = {}
    
    for pair in param_str.split(';'):
        if ':' in pair:
            key, value = pair.split(':', 1)
            params[key] = value
    
    return params
```

### Match Relocalization Parameters to Map

Updated `localize()` to use the target map's parameters:

```python
# NEW CODE (fixed)
db_params = self.extract_params_from_db(str(map_path))

reloc_params = constants.DEFAULT_PARAMS.copy()

if 'BRIEF/Bytes' in db_params:
    reloc_params['BRIEF/Bytes'] = db_params['BRIEF/Bytes']
    print(f"[RTAB-Map] Using map's BRIEF descriptor size: {db_params['BRIEF/Bytes']} bytes")

if 'Kp/DetectorStrategy' in db_params:
    reloc_params['Kp/DetectorStrategy'] = db_params['Kp/DetectorStrategy']
```

## How It Works Now

### Relocalization Flow

1. **Extract target map parameters**:
   ```
   map_session_20260209_191433_b716707e.db
   ↓
   extract_params_from_db()
   ↓
   BRIEF/Bytes: 32, Kp/DetectorStrategy: 6
   ```

2. **Override relocalization parameters**:
   ```
   DEFAULT_PARAMS (64 bytes)
   ↓
   Override with map's params (32 bytes)
   ↓
   reloc_params['BRIEF/Bytes'] = "32"
   ```

3. **Feature extraction matches**:
   ```
   Query image → BRIEF descriptor (32 bytes)
   Map database → Dictionary expects 32 bytes
   ✅ Match successful!
   ```

## Testing

Verified parameter extraction works for both databases:

```bash
iOS DB (260202-202240.db):
  BRIEF/Bytes: 64
  Kp/DetectorStrategy: 6

Our DB (map_session_20260209_191433_b716707e.db):
  BRIEF/Bytes: 32
  Kp/DetectorStrategy: 6
```

## Benefits

1. **Automatic compatibility**: No manual configuration needed
2. **Works with any map**: iOS app maps, Flutter app maps, or mixed
3. **Prevents errors**: Descriptor size always matches target map
4. **Future-proof**: New parameters can be extracted same way

## Related Files

- `be/slam_engines/rtabmap/engine.py`: Added `extract_params_from_db()` and updated `localize()`
- `be/slam_engines/rtabmap/constants.py`: DEFAULT_PARAMS (used as base, then overridden)
- `be/routes/localize.py`: Endpoint that calls `engine.localize()`

## Database Schema

RTAB-Map stores parameters in the `Info` table:

```sql
SELECT parameters FROM Info LIMIT 1;
-- Returns: "BRIEF/Bytes:32;BRISK/Octaves:3;...;Vis/MinInliers:10"
```

Format: Semicolon-separated `key:value` pairs, parsed by our extractor.

## Why Different Descriptor Sizes?

Our databases were created with different RTAB-Map configurations:

- **iOS app**: Default RTAB-Map app settings (64-byte BRIEF)
- **Flutter app**: Custom settings in `constants.py` (originally set to 64, but DB was created with 32)

The mismatch likely occurred during early development when we changed `constants.py` but kept testing with old databases.

## Prevention

To avoid future mismatches:
1. Always check `BRIEF/Bytes` in `constants.py` before creating new maps
2. Document the descriptor size in map metadata
3. Use `extract_params_from_db()` for any operation that loads an existing map
