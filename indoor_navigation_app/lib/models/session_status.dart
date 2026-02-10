// lib/models/session_status.dart
class SessionStatus {
  final String sessionId;
  final String status; // scanning | queued | processing | completed | failed
  final double progress;
  final int totalFrames;
  final int totalChunks;
  final String? mapId;
  final String? error;
  
  SessionStatus({
    required this.sessionId,
    required this.status,
    this.progress = 0.0,
    this.totalFrames = 0,
    this.totalChunks = 0,
    this.mapId,
    this.error,
  });
  
  factory SessionStatus.fromJson(Map<String, dynamic> json) {
    return SessionStatus(
      sessionId: json['session_id'],
      status: json['status'],
      progress: (json['progress'] ?? 0).toDouble(),
      totalFrames: json['total_frames'] ?? 0,
      totalChunks: json['total_chunks'] ?? 0,
      mapId: json['map_id'],
      error: json['error'],
    );
  }
  
  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';
  bool get isProcessing => status == 'processing';
}

