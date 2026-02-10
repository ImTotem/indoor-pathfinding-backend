import 'dart:typed_data';
import 'package:flutter/services.dart';

class ARCoreSharedCameraService {
  static const MethodChannel _channel = MethodChannel('arcore_shared_camera');

  bool _isInitialized = false;
  bool _isStreaming = false;

  bool get isInitialized => _isInitialized;
  bool get isStreaming => _isStreaming;

  Future<bool> initialize() async {
    try {
      final result = await _channel.invokeMethod('initialize');
      _isInitialized = result == true;
      print('[ARCoreSharedCamera] Initialized: $_isInitialized');
      return _isInitialized;
    } catch (e) {
      print('[ARCoreSharedCamera] Initialization failed: $e');
      _isInitialized = false;
      return false;
    }
  }

  Future<bool> startCamera() async {
    if (!_isInitialized) {
      print('[ARCoreSharedCamera] Not initialized');
      return false;
    }

    try {
      final result = await _channel.invokeMethod('startCamera');
      _isStreaming = result == true;
      print('[ARCoreSharedCamera] Camera started: $_isStreaming');
      return _isStreaming;
    } catch (e) {
      print('[ARCoreSharedCamera] Failed to start camera: $e');
      _isStreaming = false;
      return false;
    }
  }

  Future<Map<String, dynamic>?> captureFrame() async {
    if (!_isStreaming) {
      return null;
    }

    try {
      final result = await _channel.invokeMethod('captureFrame');
      if (result == null) {
        return null;
      }

      final data = Map<String, dynamic>.from(result);

      if (data.containsKey('jpegData')) {
        data['jpegData'] = data['jpegData'] as Uint8List;
      }
      if (data.containsKey('depthData')) {
        data['depthData'] = data['depthData'] as Uint8List;
      }
      if (data.containsKey('position')) {
        data['position'] = List<double>.from(data['position']);
      }
      if (data.containsKey('orientation')) {
        data['orientation'] = List<double>.from(data['orientation']);
      }

      return data;
    } catch (e) {
      print('[ARCoreSharedCamera] Failed to capture frame: $e');
      return null;
    }
  }

  Future<void> stopCamera() async {
    try {
      await _channel.invokeMethod('stopCamera');
      _isStreaming = false;
      print('[ARCoreSharedCamera] Camera stopped');
    } catch (e) {
      print('[ARCoreSharedCamera] Failed to stop camera: $e');
    }
  }

  Future<void> dispose() async {
    try {
      await _channel.invokeMethod('dispose');
      _isInitialized = false;
      _isStreaming = false;
      print('[ARCoreSharedCamera] Disposed');
    } catch (e) {
      print('[ARCoreSharedCamera] Failed to dispose: $e');
    }
  }
}
