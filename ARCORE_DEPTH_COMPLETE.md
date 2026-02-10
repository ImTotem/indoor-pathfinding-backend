# ARCore Depth Integration - Implementation Complete

## Summary

ARCore ML-based depth-from-motion API has been integrated end-to-end for dense 3D reconstruction in RTAB-Map RGB-D mode.

## What Was Implemented

### 1. Flutter Frontend (ARCore Depth Capture)

#### Native Android Plugin
- **File**: `indoor_navigation_app/android/app/src/main/kotlin/.../ARCoreDepthPlugin.kt`
- **Functionality**: 
  - Captures 16-bit depth maps via `frame.acquireDepthImage16Bits()`
  - Returns depth dimensions (width, height) and raw bytes
  - Handles ARCore session lifecycle (init, capture, dispose)

#### Dart Service Layer
- **File**: `indoor_navigation_app/lib/services/arcore_depth_service.dart`
- **Methods**: `initDepth()`, `captureDepth()`, `disposeDepth()`
- **Integration**: MethodChannel bridge to native plugin

#### MainActivity Registration
- **File**: `indoor_navigation_app/android/app/src/main/kotlin/.../MainActivity.kt`
- **Change**: Added `arcore_depth` channel registration

#### Data Model
- **File**: `indoor_navigation_app/lib/models/frame_data.dart`
- **Added Fields**:
  - `Uint8List? depthBytes`
  - `int? depthWidth`
  - `int? depthHeight`

#### Scanning Screen
- **File**: `indoor_navigation_app/lib/screens/scanning_screen.dart`
- **Changes**:
  - Initialize depth on scan start
  - Capture depth alongside RGB in `_processFrameInBackground()`
  - Dispose depth on scan stop

#### API Upload
- **File**: `indoor_navigation_app/lib/services/api_service.dart`
- **Change**: Modified multipart upload to include depth files as raw binary (`depth_0.raw`, etc.)

### 2. Backend (FastAPI)

#### Upload Endpoint
- **File**: `be/routes/scan.py`
- **Endpoint**: `POST /api/scan/chunk-binary`
- **Changes**:
  - Accept `depths` multipart file array
  - Parse `depth_widths` and `depth_heights` from metadata JSON
  - Pass depth data to storage manager

#### Storage Manager
- **File**: `be/storage/storage_manager.py`
- **Method**: `save_frame_binary()`
- **Changes**:
  - Create `{session}/depth/` directory
  - Decode raw depth bytes as `np.uint16`
  - Save as 16-bit PNG using OpenCV: `{frame_idx:06d}.png`
  - Record `depth_path` in chunk JSON metadata

#### Dependencies
- **File**: `be/requirements.txt`
- **Added**: `numpy`, `opencv-python`

### 3. SLAM Engine (RTAB-Map RGB-D)

#### Engine Core
- **File**: `be/slam_engines/rtabmap/engine.py`
- **Changes**:
  1. **Depth Detection**: Check if `{session}/depth/` exists and has files
  2. **images.txt Format**: 
     - RGB-only mode: `images/000000.jpg`
     - RGB-D mode: `images/000000.jpg depth/000000.png`
  3. **Parameter Injection**:
     - When `has_depth=True`, add to CLI args:
       - `-param RGBD/Enabled=true`
       - `-param Mem/DepthAsMask=false`
  4. **Mode Propagation**: Pass `has_depth` flag through `_run_rtabmap()` → `_run_docker()` / `_run_local()`

#### Constants
- **File**: `be/slam_engines/rtabmap/constants.py`
- **No changes**: RGB-D params injected dynamically, not in defaults

## Data Flow

```
1. Flutter ARCore Depth Capture
   ├─ ARCoreDepthPlugin.captureDepth() → uint16 depth image
   └─ ScanningScreen → FrameData(depthBytes, depthWidth, depthHeight)

2. Multipart Upload
   ├─ RGB JPEG: frame_0.jpg
   ├─ Depth raw: depth_0.raw (uint16 bytes)
   └─ Metadata JSON: {depth_widths: [...], depth_heights: [...]}

3. Backend Storage
   ├─ Decode depth as np.uint16
   ├─ Save as PNG: session_xyz/depth/000000.png
   └─ Update chunk JSON: {depth_path: "depth/000000.png"}

4. RTAB-Map Processing
   ├─ Detect depth/ directory
   ├─ Create images.txt with RGB-D pairs
   ├─ Pass -param RGBD/Enabled=true
   └─ Generate dense 3D map (walls/floors distinguished)
```

## Expected Output

### Monocular RGB Mode (Before)
- Sparse point cloud (~1K-5K points)
- Feature points only (corners, edges)
- No surface reconstruction

### RGB-D Mode (After)
- Dense point cloud (~100K-500K points)
- Surface reconstruction of walls, floors, objects
- Sufficient for wall/floor segmentation

## Commits

1. **86e0d14**: "Add ARCore depth integration for dense reconstruction"
   - Frontend: ARCore plugin, service, model, scanning integration
   - Backend: Upload endpoint, storage manager, depth PNG saving

2. **83fa992**: "Configure RTAB-Map for RGB-D mode with depth maps"
   - SLAM engine: Depth detection, images.txt pairing, parameter injection

## Testing Required

### 1. Depth Capture Test
- Run Flutter app on Galaxy S23
- Start scan, verify logs show: `[ARCoreDepth] Init succeeded`
- Check depth capture logs: `[Scanning] Depth captured: 160x120, 38400 bytes`

### 2. Upload Verification
- Backend logs should show: `[CHUNK-BINARY] ... depths=N`
- Check session directory: `ls be/data/sessions/{session_id}/depth/` → PNG files exist
- Verify PNG is 16-bit: `file 000000.png` → "16-bit grayscale"

### 3. SLAM Processing Test
- Backend should detect depth: `[RTAB-Map] Depth maps detected - using RGB-D mode`
- images.txt should have pairs: `cat images.txt` → `images/000000.jpg depth/000000.png`
- RTAB-Map CLI should include: `-param RGBD/Enabled=true`

### 4. Output Verification
- 3D viewer: Point cloud should be significantly denser
- Walls and floors should have visible surface coverage
- Compare sparse (RGB-only) vs dense (RGB-D) maps

## Known Limitations

### ARCore Depth API
- **Resolution**: 160×120 to 320×240 (device-dependent)
- **Quality**: ML-based, less accurate than hardware ToF
- **Latency**: 1-3 frames (~33-100ms)
- **Movement Required**: Depth-from-motion needs camera translation
- **Static Scenes**: Depth quality degrades without movement

### RTAB-Map RGB-D
- **Alignment**: Assumes RGB and depth are spatially/temporally aligned
- **Calibration**: Uses RGB camera intrinsics for depth as well
- **Scaling**: Depth in millimeters, matches RTAB-Map expectations

## Future Improvements

1. **Depth Quality Filtering**
   - Reject depth frames with high variance (noise)
   - Use confidence masks if ARCore provides them

2. **Depth Alignment**
   - Apply transform if RGB/depth cameras are offset
   - Timestamp synchronization for moving scenes

3. **Sparse-Dense Fusion**
   - Use sparse features for camera tracking (robust)
   - Use dense depth for reconstruction (detail)

4. **Wall/Floor Segmentation**
   - Post-process point cloud to extract planar surfaces
   - Label horizontal planes (floor/ceiling) vs vertical (walls)

## Status

✅ **Implementation Complete**
⏳ **Push Pending** (large sensor data files causing timeout)
🧪 **Testing Required** (on device with actual depth capture)

---

**Next Step**: Test on Samsung Galaxy S23 device to verify end-to-end depth capture → upload → SLAM → dense map generation.
