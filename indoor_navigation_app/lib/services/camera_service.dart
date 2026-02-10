// lib/services/camera_service.dart
import 'dart:async';
import 'dart:ui' show Size;
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/services.dart';
import '../models/camera_intrinsics.dart';

class CameraService {
  static const MethodChannel _channel = MethodChannel('camera_intrinsics');

  CameraController? _controller;
  CameraIntrinsics? cameraIntrinsics;
  bool _isInitialized = false;
  bool _isCapturing = false;

  // Latest frame from stream (for polling access)
  CameraImage? _latestFrame;

  // Callback for frame delivery
  Function(CameraImage image, int timestamp)? onFrameReady;

  bool get isInitialized => _isInitialized;
  bool get isCapturing => _isCapturing;

  /// Initialize camera service
  Future<void> initialize() async {
    print('[CameraService] 초기화 시작...');

    // 1. Request camera permissions
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      throw Exception('Camera permission denied');
    }

    // 2. Get available cameras
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw Exception('No cameras available');
    }

    // 3. Select back camera (primary camera for SLAM)
    final backCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    // 4. Create CameraController with highest resolution
    _controller = CameraController(
      backCamera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420, // YUV for Android
    );

    // 5. Initialize controller
    await _controller!.initialize();
    _isInitialized = true;

    print('[CameraService] ✅ 초기화 완료');
    print('[CameraService]    해상도: ${_controller!.value.previewSize}');

    // 6. Extract camera intrinsics via platform channel
    await _extractCameraIntrinsics();
  }

  /// Extract camera intrinsics from native platform
  Future<void> _extractCameraIntrinsics() async {
    try {
      print('[CameraService] 카메라 보정 정보 추출 중...');

      final Map<dynamic, dynamic> result =
          await _channel.invokeMethod('getCameraIntrinsics');
      cameraIntrinsics =
          CameraIntrinsics.fromJson(Map<String, dynamic>.from(result));

      print('[CameraService] ✅ 카메라 보정 정보: $cameraIntrinsics');
      print(
          '[CameraService]    distortion: k1=${cameraIntrinsics!.k1.toStringAsFixed(3)}, k2=${cameraIntrinsics!.k2.toStringAsFixed(3)}, p1=${cameraIntrinsics!.p1.toStringAsFixed(3)}, p2=${cameraIntrinsics!.p2.toStringAsFixed(3)}');
    } catch (e) {
      print('[CameraService] ⚠️  보정 정보 추출 실패, 기본값 사용: $e');

      cameraIntrinsics = CameraIntrinsics(
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
        width: 640,
        height: 480,
      );
    }
  }

  /// Start capturing frames
  Future<void> startCapture() async {
    if (!_isInitialized) {
      throw Exception('CameraService not initialized');
    }

    if (_isCapturing) {
      print('[CameraService] Already capturing, ignoring startCapture');
      return;
    }

    print('[CameraService] 프레임 캡처 시작...');

    await _controller!.startImageStream((CameraImage image) {
      // Store latest frame for polling access
      _latestFrame = image;

      // Manual timestamp (CameraImage has no hardware timestamp)
      final timestamp = DateTime.now().millisecondsSinceEpoch;

      // Deliver frame via callback
      onFrameReady?.call(image, timestamp);
    });

    _isCapturing = true;
    print('[CameraService] ✅ 프레임 캡처 활성화');
  }

  /// Stop capturing frames
  Future<void> stopCapture() async {
    if (!_isCapturing) {
      return;
    }

    print('[CameraService] 프레임 캡처 중지...');
    await _controller?.stopImageStream();
    _latestFrame = null;
    _isCapturing = false;
    print('[CameraService] ✅ 프레임 캡처 비활성화');
  }

  /// Dispose camera resources
  void dispose() {
    print('[CameraService] 리소스 정리...');
    stopCapture();
    _controller?.dispose();
    _controller = null;
    _isInitialized = false;
    _isCapturing = false;
    cameraIntrinsics = null;
    onFrameReady = null;
    print('[CameraService] ✅ 정리 완료');
  }

  /// Get current preview size (for intrinsics scaling)
  Size? get previewSize => _controller?.value.previewSize;

  /// Get camera controller (for CameraPreview widget)
  CameraController? get controller => _controller;

  /// Get latest frame from stream (for polling-based capture)
  CameraImage? getLatestFrame() => _latestFrame;
}
