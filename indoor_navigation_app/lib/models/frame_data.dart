// lib/models/frame_data.dart
import 'dart:typed_data';
import 'camera_intrinsics.dart';

class FrameData {
  final Uint8List imageBytes;
  final Uint8List? depthBytes;
  final int? depthWidth;
  final int? depthHeight;
  final List<double> position;
  final List<double> orientation;
  final int timestamp;
  final Map<String, dynamic>? imuData;
  final CameraIntrinsics? cameraIntrinsics;

  FrameData({
    required this.imageBytes,
    this.depthBytes,
    this.depthWidth,
    this.depthHeight,
    required this.position,
    required this.orientation,
    required this.timestamp,
    this.imuData,
    this.cameraIntrinsics,
  });

  Map<String, dynamic> toJson() => {
        'image': imageBytes,
        'position': position,
        'orientation': orientation,
        'timestamp': timestamp,
        if (imuData != null) 'imu': imuData,
        if (cameraIntrinsics != null)
          'camera_intrinsics': cameraIntrinsics!.toJson(),
        if (depthBytes != null) 'depth': depthBytes,
        if (depthWidth != null) 'depth_width': depthWidth,
        if (depthHeight != null) 'depth_height': depthHeight,
      };

  int get estimatedSize => imageBytes.length + (depthBytes?.length ?? 0);
}
