import 'package:flutter/material.dart';

class FeaturePointPainter extends CustomPainter {
  final List<double> points;
  final int pointCount;

  FeaturePointPainter({required this.points, required this.pointCount});

  @override
  void paint(Canvas canvas, Size size) {
    if (pointCount == 0) return;

    final paint = Paint()
      ..color = Colors.greenAccent.withOpacity(0.8)
      ..style = PaintingStyle.fill;

    for (int i = 0; i < pointCount && (i * 2 + 1) < points.length; i++) {
      final u = points[i * 2];
      final v = points[i * 2 + 1];
      canvas.drawCircle(
        Offset(u * size.width, v * size.height),
        3.0,
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(FeaturePointPainter oldDelegate) =>
      oldDelegate.pointCount != pointCount || oldDelegate.points != points;
}
