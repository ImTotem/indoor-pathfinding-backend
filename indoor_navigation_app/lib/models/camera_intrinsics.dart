// lib/models/camera_intrinsics.dart
class CameraIntrinsics {
  final double fx;
  final double fy;
  final double cx;
  final double cy;
  final double k1;
  final double k2;
  final double p1;
  final double p2;
  final int width;
  final int height;
  
  CameraIntrinsics({
    required this.fx,
    required this.fy,
    required this.cx,
    required this.cy,
    this.k1 = 0.0,
    this.k2 = 0.0,
    this.p1 = 0.0,
    this.p2 = 0.0,
    required this.width,
    required this.height,
  });
  
  Map<String, dynamic> toJson() => {
    'fx': fx,
    'fy': fy,
    'cx': cx,
    'cy': cy,
    'k1': k1,
    'k2': k2,
    'p1': p1,
    'p2': p2,
    'width': width,
    'height': height,
  };
  
  factory CameraIntrinsics.fromJson(Map<String, dynamic> json) {
    return CameraIntrinsics(
      fx: json['fx']?.toDouble() ?? 800.0,
      fy: json['fy']?.toDouble() ?? 800.0,
      cx: json['cx']?.toDouble() ?? 320.0,
      cy: json['cy']?.toDouble() ?? 240.0,
      k1: json['k1']?.toDouble() ?? 0.0,
      k2: json['k2']?.toDouble() ?? 0.0,
      p1: json['p1']?.toDouble() ?? 0.0,
      p2: json['p2']?.toDouble() ?? 0.0,
      width: json['width'] ?? 640,
      height: json['height'] ?? 480,
    );
  }
  
  @override
  String toString() {
    return 'CameraIntrinsics(fx: ${fx.toStringAsFixed(1)}, fy: ${fy.toStringAsFixed(1)}, '
           'cx: ${cx.toStringAsFixed(1)}, cy: ${cy.toStringAsFixed(1)})';
  }
}

