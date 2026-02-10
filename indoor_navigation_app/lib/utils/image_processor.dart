// lib/utils/image_processor.dart
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ImageProcessor {
  static Future<Uint8List?> compressImage(
    Uint8List bytes, {
    int width = 640,
    int height = 480,
    int quality = 90,
  }) async {
    try {
      final originalImage = img.decodeImage(bytes);
      if (originalImage == null) return null;
      
      final resized = img.copyResize(
        originalImage,
        width: width,
        height: height,
      );
      
      return Uint8List.fromList(img.encodeJpg(resized, quality: quality));
    } catch (e) {
      print('[ImageProcessor] 압축 실패: $e');
      return null;
    }
  }
}

