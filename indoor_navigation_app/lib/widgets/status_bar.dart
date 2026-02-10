// lib/widgets/status_bar.dart
import 'package:flutter/material.dart';
import 'connection_indicator.dart';
import 'info_row.dart';
import 'sensor_status.dart';
import '../models/camera_intrinsics.dart';

class StatusBar extends StatelessWidget {
  final bool isScanning;
  final bool isConnected;
  final bool hasCalibration;
  final CameraIntrinsics? cameraIntrinsics;
  final double currentFps;
  final int frameCount;
  final ValueNotifier<int>? uploadedFrames; // Changed to ValueNotifier
  final bool hasAccel;
  final bool hasGyro;
  final bool hasMagnet;

  const StatusBar({
    Key? key,
    required this.isScanning,
    required this.isConnected,
    required this.hasCalibration,
    this.cameraIntrinsics,
    required this.currentFps,
    required this.frameCount,
    required this.uploadedFrames,
    required this.hasAccel,
    required this.hasGyro,
    required this.hasMagnet,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: SafeArea(
        child: Container(
          padding: EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                Colors.black.withOpacity(0.7),
                Colors.transparent,
              ],
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.camera_alt, color: Colors.white, size: 24),
                  SizedBox(width: 8),
                  Text(
                    '실내 스캐닝',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Spacer(),
                  ConnectionIndicator(isConnected: isConnected),
                ],
              ),
              // ALWAYS show calibration and sensors
              SizedBox(height: 12),
              _buildCalibrationStatus(),
              SizedBox(height: 4),
              SensorStatus(
                hasAccel: hasAccel,
                hasGyro: hasGyro,
                hasMagnet: hasMagnet,
              ),
              // Show metrics ONLY when scanning
              if (isScanning) ...[
                SizedBox(height: 8),
                Divider(color: Colors.white30, thickness: 1),
                SizedBox(height: 4),
                InfoRow(
                  icon: Icons.speed,
                  label: 'FPS',
                  value: currentFps.toStringAsFixed(1),
                  color: Colors.yellow,
                ),
                SizedBox(height: 4),
                InfoRow(
                  icon: Icons.photo_camera,
                  label: '프레임',
                  value: '$frameCount',
                  color: Colors.blue,
                ),
                SizedBox(height: 4),
                ValueListenableBuilder<int>(
                  valueListenable: uploadedFrames ?? ValueNotifier<int>(0),
                  builder: (context, uploaded, child) {
                    return InfoRow(
                      icon: Icons.cloud_upload,
                      label: '전송됨',
                      value: '$uploaded',
                      color: Colors.green,
                    );
                  },
                ),
                SizedBox(height: 4),
                ValueListenableBuilder<int>(
                  valueListenable: uploadedFrames ?? ValueNotifier<int>(0),
                  builder: (context, uploaded, child) {
                    final pending = frameCount - uploaded;
                    return InfoRow(
                      icon: Icons.queue,
                      label: '대기',
                      value: '${pending > 0 ? pending : 0}',
                      color: Colors.orange,
                    );
                  },
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCalibrationStatus() {
    return Row(
      children: [
        Icon(
          hasCalibration ? Icons.check_circle : Icons.warning,
          color: hasCalibration ? Colors.green : Colors.orange,
          size: 16,
        ),
        SizedBox(width: 8),
        Text(
          '카메라 보정:',
          style: TextStyle(color: Colors.white70, fontSize: 14),
        ),
        SizedBox(width: 8),
        Text(
          hasCalibration ? '활성' : '대기 중',
          style: TextStyle(
            color: hasCalibration ? Colors.green : Colors.orange,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        if (hasCalibration && cameraIntrinsics != null) ...[
          SizedBox(width: 8),
          Text(
            '(fx:${cameraIntrinsics!.fx.toStringAsFixed(0)})',
            style: TextStyle(color: Colors.white60, fontSize: 12),
          ),
        ],
      ],
    );
  }
}
