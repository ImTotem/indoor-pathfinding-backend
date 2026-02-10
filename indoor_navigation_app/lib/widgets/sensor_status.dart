// lib/widgets/sensor_status.dart
import 'package:flutter/material.dart';

class SensorStatus extends StatelessWidget {
  final bool hasAccel;
  final bool hasGyro;
  final bool hasMagnet;
  
  const SensorStatus({
    Key? key,
    required this.hasAccel,
    required this.hasGyro,
    required this.hasMagnet,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(Icons.sensors, color: Colors.cyan, size: 16),
        SizedBox(width: 8),
        Text('IMU:', style: TextStyle(color: Colors.white70, fontSize: 14)),
        SizedBox(width: 8),
        _buildSensorChip('가속', hasAccel),
        SizedBox(width: 4),
        _buildSensorChip('자이로', hasGyro),
        SizedBox(width: 4),
        _buildSensorChip('자기', hasMagnet),
      ],
    );
  }
  
  Widget _buildSensorChip(String label, bool active) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: active ? Colors.cyan.withOpacity(0.3) : Colors.red.withOpacity(0.3),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: active ? Colors.cyan : Colors.red,
          width: 1,
        ),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: active ? Colors.cyan : Colors.red,
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}

