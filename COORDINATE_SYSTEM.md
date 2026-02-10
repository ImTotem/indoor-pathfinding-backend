# RTAB-Map Coordinate System Mappings

## Coordinate Conventions

### RTAB-Map Robot Frame (ROS REP-103)
- **X**: Forward (front of robot)
- **Y**: Left (left side of robot)
- **Z**: Up (vertical)
- **Right-handed** coordinate system

### Camera Optical Frame (OpenCV)
- **X**: Right (→)
- **Y**: Down (↓)
- **Z**: Forward (into scene)
- Used internally by RTAB-Map for camera intrinsics

### Three.js Rendering (WebGL)
- **X**: Right (→)
- **Y**: Up (↑)
- **Z**: Backward (out of screen, -Z is forward)
- **Right-handed** coordinate system

## Transformation Pipeline

### 1. ARCore → RTAB-Map Database
ARCore provides camera pose in optical frame convention. Our Flutter app captures:
- RGB image (1920×1080)
- Depth image from ARCore (160×90)
- Camera pose (position + quaternion rotation)
- Camera intrinsics

These are stored in RTAB-Map database with **no coordinate transform** - raw optical frame data.

### 2. RTAB-Map Database → Viewer
When rendering the 3D viewer, we transform from RTAB-Map coordinates to Three.js:

**Transform function**: `rt2t(x, y, z) → [x, z, y]`

This mapping:
- **X** (RTAB-Map right/forward) → **X** (Three.js right) ✅
- **Y** (RTAB-Map down/left) → **Z** (Three.js backward) 🔄
- **Z** (RTAB-Map forward/up) → **Y** (Three.js up) 🔄

### Why This Transform Works

RTAB-Map stores poses in optical frame (from ARCore):
- RTAB-Map X = Camera right
- RTAB-Map Y = Camera down
- RTAB-Map Z = Camera forward (into scene)

Three.js viewer expects:
- Three.js X = Right ✅ (matches RTAB-Map X)
- Three.js Y = Up (opposite of RTAB-Map Y down, but we want Z up)
- Three.js Z = Backward (opposite of forward, but Y becomes -Z for floor)

**Result**: The floor (Y=down in optical frame) maps to the XZ plane in Three.js, with Y pointing up.

## Testing History

### Initial Implementation (Commit ea631a8)
- Transform: `[x, -z, y]` (negated Z)
- **Issue**: Scene appeared upside-down
- User reported: "위아래가 뒤집힘" (upside-down)

### User's Fix (Manual edit)
- Transform: `[x, z, y]` (removed Z negation)
- **Result**: Correct orientation
- User confirmed: "직접 수정해보니까 알겠다" (I figured it out by testing)

## Applied Locations

The `rt2t(x, y, z)` transform is applied to ALL rendered geometry:

1. **PLY point cloud**: positions and normals
2. **Trajectory path**: keyframe positions
3. **Keyframe markers**: camera position indicators
4. **Grid**: floor reference plane
5. **Axes helper**: coordinate reference
6. **Current pose marker**: user location (if available)

This ensures the entire scene maintains consistent orientation.

## References

- **ROS REP-103**: https://www.ros.org/reps/rep-0103.html
- **RTAB-Map CameraModel**: Optical rotation transform in source code
- **Three.js Coordinate System**: Right-handed, Y-up convention
- **OpenCV Camera Model**: Right-down-forward optical frame
