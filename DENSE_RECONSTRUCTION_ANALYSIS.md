# RTAB-Map Dense Reconstruction: iOS vs Flutter App

## Issue: iOS Database Cannot Generate Dense Point Cloud

### Problem
The iOS RTAB-Map app database (`260202-202240.db`) fails to generate a dense point cloud using `rtabmap-export --cloud`, with all 545 nodes rejected:

```
Node X doesn't have depth or stereo data, empty cloud is created
```

### Root Cause

| Component | iOS RTAB-Map App | Our Flutter App | Impact |
|-----------|------------------|-----------------|--------|
| **Depth images in Data table** | **0/545 (0%)** | **11/11 (100%)** | ❌ CRITICAL |
| Depth BLOB size per node | 0 bytes | ~1.06 MB | Dense reconstruction requires depth images |
| Image BLOB size per node | ~180-230 KB | ~300-450 KB | Higher quality in our app |
| Total nodes | 545 | 11 | iOS has more coverage but no depth data |
| Feature depths (triangulated) | 8,498 features | 10,032 features | Both have sparse features, but not sufficient |

### Why iOS DB Has No Depth Data

The iOS RTAB-Map app was configured for **mono visual SLAM**:
- ARKit depth capture was either:
  - Not available on the device (older iPhone without LiDAR)
  - Not enabled in the app settings
  - Or app version didn't support ARKit depth

Even though the database has `RGBD/Enabled:true` in parameters, **no depth images were stored** in the Data table.

### Database Configuration Comparison

Both databases have identical SLAM configuration:
```
RGBD/Enabled: true
Odom/Strategy: 0 (Frame-to-Map)
Kp/DetectorStrategy: 6 (ORB)
Mem/DepthCompressionFormat: .rvl (RVL compression)
```

**Key difference**:
- `Mem/DepthAsMask`: 
  - iOS: `true` (treats depth as binary mask only)
  - Our app: `false` (uses full depth values)

### How Dense Reconstruction Works

RTAB-Map's `rtabmap-export --cloud` requires:

1. **RGB images** in Data table ✅ (both have)
2. **Depth images** in Data table ❌ (iOS missing)
3. **Camera calibration** ✅ (both have)
4. **Node poses** ✅ (both have)

The tool reconstructs dense geometry by:
- Unprojecting each depth pixel to 3D using camera intrinsics
- Transforming to world coordinates using node pose
- Assigning RGB color from corresponding RGB image pixel
- Merging overlapping depth maps from multiple keyframes

**Without depth images**, RTAB-Map can only export sparse feature points (from triangulation), which is insufficient for dense reconstruction.

### Our Flutter App Success

Our app successfully generates dense point clouds because:

1. **ARCore depth capture**: 160×90 depth images from ToF sensor or depth-from-motion
2. **Full depth storage**: Each Data row stores ~1.06 MB compressed depth
3. **RGB-D fusion**: RTAB-Map processes paired RGB + depth for dense reconstruction
4. **Result**: 167,812 colored points from just 11 keyframes

### Verification Commands

```bash
# Check depth coverage
sqlite3 be/data/maps/260202-202240.db \
  "SELECT COUNT(*) FROM Data WHERE depth IS NOT NULL;"
# Output: 0

sqlite3 be/data/maps/map_session_20260209_191433_b716707e.db \
  "SELECT COUNT(*) FROM Data WHERE depth IS NOT NULL;"
# Output: 11

# Check data sizes
sqlite3 be/data/maps/260202-202240.db \
  "SELECT id, LENGTH(image), LENGTH(depth) FROM Data LIMIT 5;"
# Output: depth column is NULL for all rows

# Attempt dense export on iOS DB
docker exec rtabmap rtabmap-export --cloud /data/maps/260202-202240.db
# Output: "Node X doesn't have depth or stereo data" for all 545 nodes
```

### Lessons Learned

1. **RGBD/Enabled != Depth Data Available**: Configuration flag doesn't guarantee data presence
2. **iOS RTAB-Map app limitations**: May not capture depth on all devices
3. **Flutter + ARCore advantage**: Programmatic control over depth capture
4. **Dense reconstruction requirement**: Must have depth images in Data table, not just triangulated features

### Solution

For dense reconstruction, you must:
- Use a device with depth sensor (LiDAR, ToF, or depth-from-motion)
- Enable depth capture in the RTAB-Map app or custom implementation
- Verify depth images are stored in the Data table (`LENGTH(depth) > 0`)
- Our Flutter app with ARCore meets all these requirements ✅

## References
- RTAB-Map database schema: `be/slam_engines/rtabmap/database_parser.py`
- Depth capture: `indoor_navigation_app/lib/services/ar_service.dart`
- PLY export: `be/slam_engines/rtabmap/engine.py` (`_run_export()`)
