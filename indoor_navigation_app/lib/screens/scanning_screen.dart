import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:typed_data';

import '../services/api_service.dart';
import '../services/streaming_uploader.dart';
import '../services/sensor_service.dart';
import '../services/arcore_shared_camera_service.dart';
import '../models/frame_data.dart';
import '../models/camera_intrinsics.dart';
import '../config/app_config.dart';
import '../config/constants.dart';
import '../widgets/status_bar.dart';
import '../widgets/crosshair_painter.dart';
import '../widgets/action_button.dart';
import '../widgets/feature_point_overlay.dart';
import 'processing_screen.dart';

class ScanningScreen extends StatefulWidget {
  @override
  _ScanningScreenState createState() => _ScanningScreenState();
}

class _ScanningScreenState extends State<ScanningScreen> {
  final ARCoreSharedCameraService _sharedCamera = ARCoreSharedCameraService();
  final SensorService _sensorService = SensorService();

  StreamingUploader? _uploader;
  Timer? _captureTimer;

  String _sessionId = '';
  bool _isScanning = false;
  bool _isPaused = false;
  bool _isConnected = true;
  bool _isCapturing = false;
  bool _cameraReady = false;

  int _frameCount = 0;
  CameraIntrinsics? _cameraIntrinsics;

  double _currentFps = 0.0;
  DateTime _lastFrameTime = DateTime.now();
  Uint8List? _previewJpeg;

  List<double> _featurePoints = [];
  int _featurePointCount = 0;

  @override
  void initState() {
    super.initState();
    _initializeAll();
  }

  Future<void> _initializeAll() async {
    _sensorService.startListening(isRecording: true);

    try {
      await _sharedCamera.initialize();
      await _sharedCamera.startCamera();
      await Future.delayed(const Duration(milliseconds: 500));

      _cameraReady = true;

      _captureTimer = Timer.periodic(
        Duration(milliseconds: AppConfig.frameIntervalMs),
        (timer) {
          if (!_isPaused) _captureFrame();
        },
      );
    } catch (e) {
      print('[Scanning] Camera init failed: $e');
    }

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          _buildCameraPreview(),
          StatusBar(
            isScanning: _isScanning,
            isConnected: _isConnected,
            hasCalibration: _cameraIntrinsics != null,
            cameraIntrinsics: _cameraIntrinsics,
            currentFps: _currentFps,
            frameCount: _frameCount,
            uploadedFrames: _uploader?.totalFramesSent,
            hasAccel: _sensorService.hasAccel,
            hasGyro: _sensorService.hasGyro,
            hasMagnet: _sensorService.hasMagnet,
          ),
          if (_featurePointCount > 0)
            Positioned.fill(
              child: RotatedBox(
                quarterTurns: 1,
                child: CustomPaint(
                  painter: FeaturePointPainter(
                    points: _featurePoints,
                    pointCount: _featurePointCount,
                  ),
                ),
              ),
            ),
          if (_isScanning && !_isPaused) _buildCrosshair(),
          _buildBottomControls(),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (_previewJpeg == null) {
      return Container(color: Colors.black);
    }
    return Positioned.fill(
      child: RotatedBox(
        quarterTurns: 1,
        child: Image.memory(
          _previewJpeg!,
          fit: BoxFit.cover,
          gaplessPlayback: true,
        ),
      ),
    );
  }

  Widget _buildCrosshair() {
    return Center(
      child: Container(
        width: AppConstants.crosshairSize,
        height: AppConstants.crosshairSize,
        child: CustomPaint(painter: CrosshairPainter()),
      ),
    );
  }

  Widget _buildBottomControls() {
    return Positioned(
      bottom: 0,
      left: 0,
      right: 0,
      child: SafeArea(
        bottom: true,
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                Colors.transparent,
                Colors.black.withOpacity(0.7),
              ],
            ),
          ),
          child: Padding(
            padding: EdgeInsets.symmetric(horizontal: 20, vertical: 20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    _getStatusMessage(),
                    style: TextStyle(color: Colors.white, fontSize: 14),
                    textAlign: TextAlign.center,
                  ),
                ),
                SizedBox(height: 20),
                _buildActionButtons(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  String _getStatusMessage() {
    if (!_cameraReady) return '카메라 초기화 중...';
    if (_isPaused) return AppConstants.msgPaused;
    if (_isScanning) return AppConstants.msgScanning;
    return AppConstants.msgReady;
  }

  Widget _buildActionButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (!_isScanning)
          ActionButton(
            icon: Icons.play_arrow,
            label: '스캔 시작',
            color: Color(AppConstants.colorGreen),
            onPressed: _cameraReady ? _startScanning : null,
          ),
        if (_isScanning) ...[
          ActionButton(
            icon: _isPaused ? Icons.play_arrow : Icons.pause,
            label: _isPaused ? '재개' : '일시정지',
            color: Color(AppConstants.colorOrange),
            onPressed: _togglePause,
          ),
          SizedBox(width: 16),
          ActionButton(
            icon: Icons.stop,
            label: '완료',
            color: Color(AppConstants.colorRed),
            onPressed: _stopScanning,
          ),
        ],
      ],
    );
  }

  Future<void> _startScanning() async {
    try {
      _sessionId = await ApiService.startScan();
      _uploader = StreamingUploader(sessionId: _sessionId);
      _sensorService.clearBuffer();

      setState(() {
        _isScanning = true;
        _isPaused = false;
        _frameCount = 0;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              Icon(Icons.check_circle, color: Colors.white),
              SizedBox(width: 8),
              Text('스캔 시작 - 천천히 움직이며 주변을 촬영하세요'),
            ],
          ),
          backgroundColor: Color(AppConstants.colorGreen),
          behavior: SnackBarBehavior.floating,
        ),
      );
    } catch (e) {
      print('[Scanning] Start failed: $e');
      setState(() {
        _isConnected = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${AppConstants.errStartFailed}: $e'),
          backgroundColor: Color(AppConstants.colorRed),
          duration: Duration(seconds: 5),
        ),
      );
    }
  }

  void _togglePause() {
    setState(() {
      _isPaused = !_isPaused;
    });
  }

  Future<void> _captureFrame() async {
    if (_isPaused || _isCapturing) return;
    _isCapturing = true;

    try {
      final frameData = await _sharedCamera.captureFrame();
      if (frameData == null) return;

      _extractIntrinsicsOnce(frameData);

      final jpegBytes = frameData['jpegData'] as Uint8List?;
      if (jpegBytes != null) {
        final rawPoints = frameData['featurePoints'];
        setState(() {
          _previewJpeg = jpegBytes;
          if (rawPoints != null) {
            _featurePoints = List<double>.from(rawPoints as List);
            _featurePointCount = (frameData['featurePointCount'] as int?) ?? 0;
          } else {
            _featurePointCount = 0;
          }
        });

        if (_isScanning) {
          final now = DateTime.now();
          final delta = now.difference(_lastFrameTime).inMilliseconds;
          if (delta > 0) {
            _currentFps = 1000.0 / delta;
          }
          _lastFrameTime = now;
          _frameCount++;
          _processFrame(jpegBytes, frameData);
        }
      }
    } catch (e) {
      print('[Scanning] Frame capture error: $e');
    } finally {
      _isCapturing = false;
    }
  }

  void _extractIntrinsicsOnce(Map<String, dynamic> frameData) {
    if (_cameraIntrinsics != null) return;

    final fx = frameData['focalLengthX'];
    final fy = frameData['focalLengthY'];
    final cx = frameData['principalPointX'];
    final cy = frameData['principalPointY'];

    if (fx != null && fy != null && cx != null && cy != null) {
      _cameraIntrinsics = CameraIntrinsics(
        fx: (fx as num).toDouble(),
        fy: (fy as num).toDouble(),
        cx: (cx as num).toDouble(),
        cy: (cy as num).toDouble(),
        width: (frameData['imageWidth'] as int?) ?? 1920,
        height: (frameData['imageHeight'] as int?) ?? 1080,
      );
      setState(() {});
    }
  }

  void _processFrame(Uint8List jpegBytes, Map<String, dynamic> capturedFrame) {
    // wall-clock 기준으로 센서 매칭 (ARCore 모노토닉 → wall-clock 직접 비교 불가)
    final wallClockMs = DateTime.now().millisecondsSinceEpoch;
    final imuData = _sensorService.getIMUDataForTimestamp(wallClockMs);
    _sensorService.trimBuffer(); // 오래된 센서 데이터 정리

    final depthBytes = capturedFrame['depthData'] as Uint8List?;
    final depthWidth = capturedFrame['depthWidth'] as int?;
    final depthHeight = capturedFrame['depthHeight'] as int?;
    final position =
        (capturedFrame['position'] as List?)?.cast<double>() ?? [0.0, 0.0, 0.0];
    final orientation =
        (capturedFrame['orientation'] as List?)?.cast<double>() ??
            [0.0, 0.0, 0.0, 1.0];

    final frameData = FrameData(
      imageBytes: jpegBytes,
      depthBytes: depthBytes,
      depthWidth: depthWidth,
      depthHeight: depthHeight,
      position: position,
      orientation: orientation,
      timestamp:
          (capturedFrame['timestamp'] as int) ~/ 1000000, // ARCore ns → ms
      imuData: imuData,
      cameraIntrinsics: _cameraIntrinsics,
    );

    _uploader?.addFrame(frameData);
  }

  Future<void> _stopScanning() async {
    setState(() {
      _isScanning = false;
      _isPaused = false;
    });

    final uploader = _uploader;
    if (uploader == null) return;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => Center(
        child: Container(
          padding: EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text(AppConstants.msgUploading, style: TextStyle(fontSize: 16)),
              SizedBox(height: 8),
              ValueListenableBuilder<int>(
                valueListenable: uploader.totalFramesSent,
                builder: (context, sent, child) {
                  final total = uploader.totalFramesQueued;
                  final remaining = total - sent;
                  return Text('$sent/$total 전송 완료 ($remaining개 남음)',
                      style: TextStyle(fontSize: 12, color: Colors.grey));
                },
              ),
            ],
          ),
        ),
      ),
    );

    try {
      await uploader.finish();

      Navigator.pop(context);
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => ProcessingScreen(sessionId: _sessionId),
        ),
      );
    } catch (e) {
      Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${AppConstants.errFinishFailed}: $e'),
          backgroundColor: Color(AppConstants.colorRed),
        ),
      );
    }
  }

  @override
  void dispose() {
    _captureTimer?.cancel();
    _uploader?.dispose();
    _sharedCamera.stopCamera();
    _sharedCamera.dispose();
    _sensorService.dispose();
    super.dispose();
  }
}
