// lib/utils/frame_quality_checker.dart
import 'dart:typed_data';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:vector_math/vector_math_64.dart' as vector;
import '../config/app_config.dart';

/// 프레임 품질 검사 결과
class FrameQualityResult {
  final bool isAcceptable;
  final double blurScore;
  final double movementDistance;
  final double rotationAngle;
  final double movementSpeed;
  final String? rejectionReason;

  FrameQualityResult({
    required this.isAcceptable,
    required this.blurScore,
    required this.movementDistance,
    required this.rotationAngle,
    required this.movementSpeed,
    this.rejectionReason,
  });
}

/// 움직임 상태
enum MovementStatus {
  tooSlow,    // 너무 느림 - 더 움직여야 함
  good,       // 적절함
  tooFast,    // 너무 빠름 - 천천히 움직여야 함
  blurry,     // 흔들림 감지
}

class FrameQualityChecker {
  vector.Vector3? _lastPosition;
  vector.Quaternion? _lastOrientation;
  DateTime? _lastCaptureTime;
  int _framesSinceLastKeyframe = 0;

  /// 블러 점수 계산 (Laplacian variance) - 샘플링으로 최적화
  /// 높을수록 선명함
  static double calculateBlurScore(Uint8List imageBytes) {
    try {
      final image = img.decodeImage(imageBytes);
      if (image == null) return 100.0; // 실패시 통과

      // 성능을 위해 이미지 축소
      final small = img.copyResize(image, width: 160, height: 120);
      final grayscale = img.grayscale(small);

      double sum = 0;
      double sumSq = 0;
      int count = 0;

      // 샘플링 (매 2픽셀마다)
      for (int y = 2; y < grayscale.height - 2; y += 2) {
        for (int x = 2; x < grayscale.width - 2; x += 2) {
          final center = grayscale.getPixel(x, y).r.toDouble();
          final left = grayscale.getPixel(x - 1, y).r.toDouble();
          final right = grayscale.getPixel(x + 1, y).r.toDouble();
          final up = grayscale.getPixel(x, y - 1).r.toDouble();
          final down = grayscale.getPixel(x, y + 1).r.toDouble();

          // 간단한 Laplacian (4방향)
          final laplacian = (4 * center - left - right - up - down).abs();
          sum += laplacian;
          sumSq += laplacian * laplacian;
          count++;
        }
      }

      if (count == 0) return 100.0;

      final mean = sum / count;
      final variance = (sumSq / count) - (mean * mean);

      return variance;
    } catch (e) {
      print('[FrameQualityChecker] 블러 점수 계산 실패: $e');
      return 100.0; // 실패시 통과
    }
  }

  /// 이동 거리 계산
  double calculateMovementDistance(vector.Vector3 currentPosition) {
    if (_lastPosition == null) {
      return 0.0;
    }
    return (currentPosition - _lastPosition!).length;
  }

  /// 회전 각도 계산 (도 단위)
  double calculateRotationAngle(vector.Quaternion currentOrientation) {
    if (_lastOrientation == null) {
      return 0.0;
    }

    // 쿼터니언 차이로 회전 각도 계산
    final diff = currentOrientation * _lastOrientation!.conjugated();
    final angle = 2 * acos(diff.w.clamp(-1.0, 1.0)) * (180 / pi);

    return angle.abs();
  }

  /// 이동 속도 계산 (m/s)
  double calculateMovementSpeed(vector.Vector3 currentPosition) {
    if (_lastPosition == null || _lastCaptureTime == null) {
      return 0.0;
    }

    final distance = calculateMovementDistance(currentPosition);
    final timeDelta = DateTime.now().difference(_lastCaptureTime!).inMilliseconds / 1000.0;

    if (timeDelta <= 0) return 0.0;
    return distance / timeDelta;
  }

  /// 움직임 상태 확인
  MovementStatus getMovementStatus(
    vector.Vector3 currentPosition,
    vector.Quaternion currentOrientation,
    double? blurScore,
  ) {
    // 블러 체크
    if (blurScore != null && blurScore < AppConfig.blurThreshold) {
      return MovementStatus.blurry;
    }

    final speed = calculateMovementSpeed(currentPosition);

    // 속도 체크
    if (speed > AppConfig.maxMovementSpeed) {
      return MovementStatus.tooFast;
    }

    final distance = calculateMovementDistance(currentPosition);
    final rotation = calculateRotationAngle(currentOrientation);

    // 충분히 움직였는지 체크
    if (distance < AppConfig.minMovementDistance &&
        rotation < AppConfig.minRotationAngle) {
      return MovementStatus.tooSlow;
    }

    return MovementStatus.good;
  }

  /// 프레임 품질 검사
  FrameQualityResult checkFrameQuality(
    Uint8List? imageBytes,
    vector.Vector3 currentPosition,
    vector.Quaternion currentOrientation,
  ) {
    final distance = calculateMovementDistance(currentPosition);
    final rotation = calculateRotationAngle(currentOrientation);
    final speed = calculateMovementSpeed(currentPosition);

    String? rejectionReason;
    bool isAcceptable = true;

    // 첫 프레임은 무조건 통과
    if (_lastPosition == null) {
      return FrameQualityResult(
        isAcceptable: true,
        blurScore: 100.0,
        movementDistance: 0,
        rotationAngle: 0,
        movementSpeed: 0,
        rejectionReason: null,
      );
    }

    // 속도 체크 (너무 빠르면 거부)
    if (speed > AppConfig.maxMovementSpeed) {
      rejectionReason = '너무 빠름';
      isAcceptable = false;
    }

    // 최소 움직임 체크 (거리 OR 회전 중 하나만 충족하면 OK)
    if (distance < AppConfig.minMovementDistance &&
        rotation < AppConfig.minRotationAngle) {
      rejectionReason = '움직임 부족';
      isAcceptable = false;
    }

    // 최소 프레임 간격 체크
    _framesSinceLastKeyframe++;
    if (_framesSinceLastKeyframe < AppConfig.minFramesBetweenKeyframes) {
      if (isAcceptable) {
        rejectionReason = '간격 부족';
        isAcceptable = false;
      }
    }

    // 블러 체크는 성능상 스킵 (나머지 조건 통과한 경우만)
    double blurScore = 100.0;
    // if (isAcceptable && imageBytes != null) {
    //   blurScore = calculateBlurScore(imageBytes);
    //   if (blurScore < AppConfig.blurThreshold) {
    //     rejectionReason = '흐림';
    //     isAcceptable = false;
    //   }
    // }

    return FrameQualityResult(
      isAcceptable: isAcceptable,
      blurScore: blurScore,
      movementDistance: distance,
      rotationAngle: rotation,
      movementSpeed: speed,
      rejectionReason: rejectionReason,
    );
  }

  /// 마지막 캡처 위치 업데이트
  void updateLastCapture(
    vector.Vector3 position,
    vector.Quaternion orientation,
  ) {
    _lastPosition = position.clone();
    _lastOrientation = orientation.clone();
    _lastCaptureTime = DateTime.now();
    _framesSinceLastKeyframe = 0;
  }

  /// 리셋
  void reset() {
    _lastPosition = null;
    _lastOrientation = null;
    _lastCaptureTime = null;
    _framesSinceLastKeyframe = 0;
  }
}
