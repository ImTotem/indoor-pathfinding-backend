import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/frame_data.dart';
import '../config/app_config.dart';
import 'api_service.dart';

class StreamingUploader {
  final String sessionId;

  /// Unbounded frame buffer — never drops frames.
  final List<FrameData> _frameBuffer = [];
  int _chunkIndex = 0;
  bool _flushing = false;

  /// Max concurrent upload requests.
  static const int _maxConcurrent = 3;
  int _activeUploads = 0;

  // ValueNotifier for reactive UI updates
  final ValueNotifier<int> totalFramesSent = ValueNotifier<int>(0);

  /// Total frames accepted (never dropped).
  int get totalFramesQueued => _totalQueued;
  int _totalQueued = 0;

  StreamingUploader({required this.sessionId});

  /// Add frame — NEVER drops. Triggers background flush when chunk is ready.
  Future<void> addFrame(FrameData frame) async {
    _frameBuffer.add(frame);
    _totalQueued++;

    // Trigger non-blocking flush when chunk is ready
    if (_frameBuffer.length >= AppConfig.chunkSize) {
      _tryFlush();
    }
  }

  /// Non-blocking: kicks off a flush if not already at max concurrency.
  void _tryFlush() {
    if (_frameBuffer.isEmpty) return;
    if (_activeUploads >= _maxConcurrent)
      return; // backpressure at network level

    // Only send full chunks during normal operation
    // Remaining frames will be sent in finish()
    if (_frameBuffer.length < AppConfig.chunkSize) return;

    // Take a full chunk from the front of the buffer
    final chunk = _frameBuffer.sublist(0, AppConfig.chunkSize);
    _frameBuffer.removeRange(0, AppConfig.chunkSize);
    final idx = _chunkIndex++;

    _activeUploads++;
    _uploadChunk(idx, chunk);
  }

  /// Fire-and-forget upload with retry.
  void _uploadChunk(int chunkIdx, List<FrameData> frames) async {
    const maxRetries = 2;
    for (int attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        final success = await ApiService.uploadChunkBinary(
          sessionId: sessionId,
          chunkIndex: chunkIdx,
          frames: frames,
        );

        if (success) {
          totalFramesSent.value += frames.length;
          print('✅ 청크 $chunkIdx 전송 완료 (총 ${totalFramesSent.value} 프레임)');
          break;
        } else {
          print('❌ 청크 $chunkIdx 전송 실패 (시도 ${attempt + 1}/${maxRetries + 1})');
          if (attempt < maxRetries) {
            await Future.delayed(Duration(milliseconds: 300 * (attempt + 1)));
          }
        }
      } catch (e) {
        print('❌ 청크 $chunkIdx 전송 에러 (시도 ${attempt + 1}): $e');
        if (attempt < maxRetries) {
          await Future.delayed(Duration(milliseconds: 300 * (attempt + 1)));
        }
      }
    }

    _activeUploads--;
    // Kick off next waiting chunk
    _tryFlush();
  }

  /// Flush all remaining frames. Called on scan stop.
  Future<void> finish() async {
    // Flush any remaining frames in buffer
    while (_frameBuffer.isNotEmpty) {
      final chunkLen = _frameBuffer.length < AppConfig.chunkSize
          ? _frameBuffer.length
          : AppConfig.chunkSize;
      final chunk = _frameBuffer.sublist(0, chunkLen);
      _frameBuffer.removeRange(0, chunkLen);
      final idx = _chunkIndex++;

      try {
        final success = await ApiService.uploadChunkBinary(
          sessionId: sessionId,
          chunkIndex: idx,
          frames: chunk,
        );
        if (success) {
          totalFramesSent.value += chunk.length;
          print('✅ 청크 $idx 전송 완료 (총 ${totalFramesSent.value} 프레임)');
        }
      } catch (e) {
        print('❌ finish 중 청크 $idx 전송 실패: $e');
      }
    }

    // Wait for in-flight uploads to complete
    while (_activeUploads > 0) {
      await Future.delayed(const Duration(milliseconds: 100));
    }

    // 서버에 완료 신호
    await ApiService.finishScan(sessionId);

    print('스캔 완료: 총 ${totalFramesSent.value}/$_totalQueued 프레임 전송됨');
  }

  void dispose() {
    totalFramesSent.dispose();
  }
}
