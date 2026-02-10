import 'dart:typed_data';
import 'package:flutter/services.dart';

class ARCoreDepthService {
  static const MethodChannel _channel = MethodChannel('arcore_depth');

  /// Initialize ARCore depth session
  /// Returns true if initialization succeeded
  Future<bool> initDepth() async {
    try {
      final result = await _channel.invokeMethod('initDepth');
      return result ?? false;
    } catch (e) {
      print('[ARCoreDepth] Init failed: $e');
      return false;
    }
  }

  /// Capture depth image
  /// Returns map with 'width', 'height', 'timestamp', 'data' (Uint8List)
  /// Returns null if depth not available
  Future<Map<String, dynamic>?> captureDepth() async {
    try {
      final result = await _channel.invokeMethod('captureDepth');
      if (result == null) return null;

      final map = Map<String, dynamic>.from(result);
      // Convert ByteData to Uint8List if needed
      if (map['data'] is ByteData) {
        final byteData = map['data'] as ByteData;
        map['data'] = byteData.buffer.asUint8List();
      }

      return map;
    } catch (e) {
      print('[ARCoreDepth] Capture failed: $e');
      return null;
    }
  }

  /// Dispose ARCore depth session
  Future<void> disposeDepth() async {
    try {
      await _channel.invokeMethod('disposeDepth');
    } catch (e) {
      print('[ARCoreDepth] Dispose failed: $e');
    }
  }
}
