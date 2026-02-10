// lib/services/sensor_service.dart
import 'dart:async';
import 'package:sensors_plus/sensors_plus.dart';

class SensorService {
  StreamSubscription<AccelerometerEvent>? _accelSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroSubscription;
  StreamSubscription<MagnetometerEvent>? _magnetSubscription;

  AccelerometerEvent? latestAccel;
  GyroscopeEvent? latestGyro;
  MagnetometerEvent? latestMagnet;

  List<Map<String, dynamic>> sensorBuffer = [];

  bool get hasAccel => latestAccel != null;
  bool get hasGyro => latestGyro != null;
  bool get hasMagnet => latestMagnet != null;

  void startListening({bool isRecording = false}) {
    _accelSubscription = accelerometerEventStream(
      samplingPeriod: SensorInterval.normalInterval,
    ).listen((AccelerometerEvent event) {
      latestAccel = event;

      if (isRecording) {
        sensorBuffer.add({
          'type': 'accel',
          'timestamp': DateTime.now().millisecondsSinceEpoch,
          'x': event.x,
          'y': event.y,
          'z': event.z,
        });
      }
    });

    _gyroSubscription = gyroscopeEventStream(
      samplingPeriod: SensorInterval.normalInterval,
    ).listen((GyroscopeEvent event) {
      latestGyro = event;

      if (isRecording) {
        sensorBuffer.add({
          'type': 'gyro',
          'timestamp': DateTime.now().millisecondsSinceEpoch,
          'x': event.x,
          'y': event.y,
          'z': event.z,
        });
      }
    });

    _magnetSubscription = magnetometerEventStream(
      samplingPeriod: SensorInterval.normalInterval,
    ).listen((MagnetometerEvent event) {
      latestMagnet = event;

      if (isRecording) {
        sensorBuffer.add({
          'type': 'magnet',
          'timestamp': DateTime.now().millisecondsSinceEpoch,
          'x': event.x,
          'y': event.y,
          'z': event.z,
        });
      }
    });
  }

  Map<String, dynamic> getIMUData() {
    return {
      'latest_accel': latestAccel != null
          ? {
              'x': latestAccel!.x,
              'y': latestAccel!.y,
              'z': latestAccel!.z,
            }
          : null,
      'latest_gyro': latestGyro != null
          ? {
              'x': latestGyro!.x,
              'y': latestGyro!.y,
              'z': latestGyro!.z,
            }
          : null,
      'latest_magnet': latestMagnet != null
          ? {
              'x': latestMagnet!.x,
              'y': latestMagnet!.y,
              'z': latestMagnet!.z,
            }
          : null,
      'num_samples': sensorBuffer.length,
    };
  }

  /// 주어진 wall-clock 타임스탬프(ms)에 가장 가까운 센서 샘플 매칭
  /// ARCore timestamp은 모노토닉이라 직접 비교 불가 → wall-clock 기준 매칭
  Map<String, dynamic> getIMUDataForTimestamp(int wallClockMs) {
    Map<String, dynamic>? closestAccel;
    Map<String, dynamic>? closestGyro;
    Map<String, dynamic>? closestMagnet;
    int bestAccelDiff = 999999999;
    int bestGyroDiff = 999999999;
    int bestMagnetDiff = 999999999;

    for (final sample in sensorBuffer) {
      final sampleTs = sample['timestamp'] as int;
      final diff = (sampleTs - wallClockMs).abs();
      final type = sample['type'] as String;

      if (type == 'accel' && diff < bestAccelDiff) {
        bestAccelDiff = diff;
        closestAccel = sample;
      } else if (type == 'gyro' && diff < bestGyroDiff) {
        bestGyroDiff = diff;
        closestGyro = sample;
      } else if (type == 'magnet' && diff < bestMagnetDiff) {
        bestMagnetDiff = diff;
        closestMagnet = sample;
      }
    }

    // 50ms 이내 매칭만 유효
    const maxDiffMs = 50;

    return {
      'accel': (closestAccel != null && bestAccelDiff <= maxDiffMs)
          ? {
              'x': closestAccel['x'],
              'y': closestAccel['y'],
              'z': closestAccel['z'],
              'dt_ms': bestAccelDiff
            }
          : null,
      'gyro': (closestGyro != null && bestGyroDiff <= maxDiffMs)
          ? {
              'x': closestGyro['x'],
              'y': closestGyro['y'],
              'z': closestGyro['z'],
              'dt_ms': bestGyroDiff
            }
          : null,
      'magnet': (closestMagnet != null && bestMagnetDiff <= maxDiffMs)
          ? {
              'x': closestMagnet['x'],
              'y': closestMagnet['y'],
              'z': closestMagnet['z'],
              'dt_ms': bestMagnetDiff
            }
          : null,
      'num_samples': sensorBuffer.length,
    };
  }

  /// 오래된 센서 버퍼 정리 (메모리 관리)
  void trimBuffer({int maxAge = 5000}) {
    final cutoff = DateTime.now().millisecondsSinceEpoch - maxAge;
    sensorBuffer.removeWhere((s) => (s['timestamp'] as int) < cutoff);
  }

  void clearBuffer() {
    sensorBuffer.clear();
  }

  void dispose() {
    _accelSubscription?.cancel();
    _gyroSubscription?.cancel();
    _magnetSubscription?.cancel();
  }
}
