// lib/services/ar_service.dart
import 'dart:async';
import 'dart:math';
import 'package:ar_flutter_plugin_2/ar_flutter_plugin.dart';
import 'package:ar_flutter_plugin_2/managers/ar_anchor_manager.dart';
import 'package:ar_flutter_plugin_2/managers/ar_location_manager.dart';
import 'package:ar_flutter_plugin_2/managers/ar_object_manager.dart';
import 'package:ar_flutter_plugin_2/managers/ar_session_manager.dart';
import 'package:ar_flutter_plugin_2/models/ar_hittest_result.dart';
import 'package:vector_math/vector_math_64.dart' as vector;
import 'package:camera_intrinsics/camera_intrinsics.dart' as ci;
import '../models/camera_intrinsics.dart';

class ARService {
  ARSessionManager? _sessionManager;
  ARObjectManager? _objectManager;
  ARAnchorManager? _anchorManager;
  ARLocationManager? _locationManager;

  vector.Vector3 currentPosition = vector.Vector3(0, 0, 0);
  vector.Quaternion currentOrientation = vector.Quaternion.identity();
  CameraIntrinsics? cameraIntrinsics;

  // 포즈 추적 상태
  bool _isTracking = false;
  int _trackingQuality = 0; // 0: 없음, 1: 제한적, 2: 정상
  DateTime _lastPoseUpdate = DateTime.now();

  // 콜백
  Function(vector.Vector3, vector.Quaternion)? onPoseUpdated;

  bool get hasCalibration => cameraIntrinsics != null;
  bool get isTracking => _isTracking;
  int get trackingQuality => _trackingQuality;

  /// 마지막 포즈 업데이트로부터 경과 시간 (ms)
  int get timeSinceLastPoseUpdate =>
      DateTime.now().difference(_lastPoseUpdate).inMilliseconds;

  void initialize(
    ARSessionManager sessionManager,
    ARObjectManager objectManager,
    ARAnchorManager anchorManager,
    ARLocationManager locationManager,
  ) {
    _sessionManager = sessionManager;
    _objectManager = objectManager;
    _anchorManager = anchorManager;
    _locationManager = locationManager;

    // AR 세션 초기화
    _sessionManager!.onInitialize(
      showFeaturePoints:
          false, // Prevents UI overlays in captured frames for SLAM
      showPlanes: false, // 성능을 위해 비활성화
      showWorldOrigin: false,
      handleTaps: true,
    );

    // 카메라 보정 정보 추출
    _extractCameraIntrinsics();

    print('[ARService] 초기화 완료');
  }

  Future<void> _extractCameraIntrinsics() async {
    // ARCore에서 실제 카메라 보정 정보 가져오기
    try {
      print('[ARService] 카메라 보정 정보 추출 중...');

      final response = await ci.CameraIntrinsics.getIntrinsics();
      final data = response.intrinsics;

      final focalLength = data['focalLength'] as List<dynamic>;
      final principalPoint = data['principalPoint'] as List<dynamic>;
      final imageDimensions = data['imageDimensions'] as List<dynamic>;

      cameraIntrinsics = CameraIntrinsics(
        fx: focalLength[0].toDouble(),
        fy: focalLength[1].toDouble(),
        cx: principalPoint[0].toDouble(),
        cy: principalPoint[1].toDouble(),
        width: imageDimensions[0] as int,
        height: imageDimensions[1] as int,
      );

      print('[ARService] ✅ 실제 카메라 보정 정보: $cameraIntrinsics');
      print('[ARService]    캐시됨: ${response.isCached}');
    } catch (e) {
      print('[ARService] ⚠️  카메라 보정 추출 실패, 기본값 사용: $e');

      // Fallback: 기본값 사용 (에러 시에만)
      cameraIntrinsics = CameraIntrinsics(
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
        width: 640,
        height: 480,
      );
    }
  }

  /// 평면 탭에서 포즈 업데이트
  void updatePoseFromHit(ARHitTestResult hit) {
    final transform = hit.worldTransform;

    currentPosition = vector.Vector3(
      transform.getColumn(3).x,
      transform.getColumn(3).y,
      transform.getColumn(3).z,
    );

    currentOrientation = _matrixToQuaternion(transform);

    _isTracking = true;
    _trackingQuality = 2;
    _lastPoseUpdate = DateTime.now();

    print(
        '[ARService] Pose: pos=(${currentPosition.x.toStringAsFixed(3)}, ${currentPosition.y.toStringAsFixed(3)}, ${currentPosition.z.toStringAsFixed(3)}), '
        'quat=(${currentOrientation.x.toStringAsFixed(3)}, ${currentOrientation.y.toStringAsFixed(3)}, '
        '${currentOrientation.z.toStringAsFixed(3)}, ${currentOrientation.w.toStringAsFixed(3)})');

    onPoseUpdated?.call(currentPosition, currentOrientation);
  }

  /// 센서 데이터로 포즈 보간 (IMU 기반)
  void updatePoseFromIMU(
    double accelX,
    double accelY,
    double accelZ,
    double gyroX,
    double gyroY,
    double gyroZ,
    double dt,
  ) {
    if (!_isTracking) return;

    // 간단한 적분으로 회전 업데이트 (gyro)
    final deltaRotation = vector.Quaternion.euler(
      gyroX * dt,
      gyroY * dt,
      gyroZ * dt,
    );
    currentOrientation = currentOrientation * deltaRotation;
    currentOrientation.normalize();

    // 가속도에서 중력 제거 후 위치 업데이트는 복잡하므로 생략
    // (드리프트가 심해서 실용적이지 않음)

    _lastPoseUpdate = DateTime.now();
  }

  /// 4x4 변환 행렬에서 쿼터니언 추출
  vector.Quaternion _matrixToQuaternion(vector.Matrix4 m) {
    // 3x3 회전 행렬 추출 (column-major order)
    final r00 = m.entry(0, 0), r01 = m.entry(0, 1), r02 = m.entry(0, 2);
    final r10 = m.entry(1, 0), r11 = m.entry(1, 1), r12 = m.entry(1, 2);
    final r20 = m.entry(2, 0), r21 = m.entry(2, 1), r22 = m.entry(2, 2);

    // Shepperd's method
    final trace = r00 + r11 + r22;
    double w, x, y, z;

    if (trace > 0) {
      final s = sqrt(trace + 1.0) * 2;
      w = 0.25 * s;
      x = (r21 - r12) / s;
      y = (r02 - r20) / s;
      z = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
      final s = sqrt(1.0 + r00 - r11 - r22) * 2;
      w = (r21 - r12) / s;
      x = 0.25 * s;
      y = (r01 + r10) / s;
      z = (r02 + r20) / s;
    } else if (r11 > r22) {
      final s = sqrt(1.0 + r11 - r00 - r22) * 2;
      w = (r02 - r20) / s;
      x = (r01 + r10) / s;
      y = 0.25 * s;
      z = (r12 + r21) / s;
    } else {
      final s = sqrt(1.0 + r22 - r00 - r11) * 2;
      w = (r10 - r01) / s;
      x = (r02 + r20) / s;
      y = (r12 + r21) / s;
      z = 0.25 * s;
    }

    return vector.Quaternion(x, y, z, w);
  }

  /// 추적 상태 리셋
  void resetTracking() {
    currentPosition = vector.Vector3(0, 0, 0);
    currentOrientation = vector.Quaternion.identity();
    _isTracking = false;
    _trackingQuality = 0;
    print('[ARService] 추적 리셋');
  }

  void dispose() {
    _sessionManager?.dispose();
  }
}
