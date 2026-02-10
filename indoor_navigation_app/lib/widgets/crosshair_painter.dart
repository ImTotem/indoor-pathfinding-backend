// lib/widgets/crosshair_painter.dart
import 'package:flutter/material.dart';

class CrosshairPainter extends CustomPainter {
  final Color color;

  CrosshairPainter({this.color = Colors.white});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final center = Offset(size.width / 2, size.height / 2);

    // 외부 원
    canvas.drawCircle(center, size.width / 2, paint);

    // 십자선
    canvas.drawLine(
      Offset(center.dx, 0),
      Offset(center.dx, 12),
      paint,
    );
    canvas.drawLine(
      Offset(center.dx, size.height - 12),
      Offset(center.dx, size.height),
      paint,
    );
    canvas.drawLine(
      Offset(0, center.dy),
      Offset(12, center.dy),
      paint,
    );
    canvas.drawLine(
      Offset(size.width - 12, center.dy),
      Offset(size.width, center.dy),
      paint,
    );

    // 중앙 점 (상태에 맞는 색상)
    canvas.drawCircle(center, 3, Paint()..color = color);
  }

  @override
  bool shouldRepaint(CrosshairPainter oldDelegate) => oldDelegate.color != color;
}

