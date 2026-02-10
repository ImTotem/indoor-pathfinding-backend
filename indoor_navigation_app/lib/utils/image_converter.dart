// lib/utils/image_converter.dart
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

class ImageConverter {
  /// Convert CameraImage to JPEG bytes
  ///
  /// Handles platform-specific formats:
  /// - Android: YUV420 (requires manual YUV→RGB conversion)
  /// - iOS: BGRA8888 (direct conversion)
  static Future<Uint8List> convertCameraImageToJpeg(
    CameraImage image, {
    int quality = 90,
  }) async {
    img.Image? rgbImage;

    if (image.format.group == ImageFormatGroup.yuv420) {
      rgbImage = _convertYUV420ToRGB(image);
    } else if (image.format.group == ImageFormatGroup.bgra8888) {
      rgbImage = _convertBGRA8888ToRGB(image);
    } else {
      throw UnsupportedError('Unsupported image format: ${image.format.group}');
    }

    final jpegBytes = img.encodeJpg(rgbImage, quality: quality);
    return Uint8List.fromList(jpegBytes);
  }

  /// Convert YUV420 to RGB (Android)
  ///
  /// CRITICAL: Uses bytesPerRow instead of width to handle padding
  static img.Image _convertYUV420ToRGB(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    final int yRowStride = image.planes[0].bytesPerRow;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    final int yMaxIndex = image.planes[0].bytes.length;
    final int uvMaxIndex =
        image.planes[1].bytes.length < image.planes[2].bytes.length
            ? image.planes[1].bytes.length
            : image.planes[2].bytes.length;

    final rgbImage = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yRowStride + x;
        final int uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;

        int yValue = 0;
        if (yIndex < yMaxIndex) {
          yValue = image.planes[0].bytes[yIndex];
        } else {
          yValue = 128;
        }

        // Bounds check for UV planes to prevent RangeError
        int uValue, vValue;
        if (uvIndex < uvMaxIndex) {
          uValue = image.planes[1].bytes[uvIndex];
          vValue = image.planes[2].bytes[uvIndex];
        } else {
          // Fallback to neutral gray (128) if out of bounds
          uValue = 128;
          vValue = 128;
        }

        final int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
        final int g =
            (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
                .round()
                .clamp(0, 255);
        final int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);

        rgbImage.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return rgbImage;
  }

  /// Convert BGRA8888 to RGB (iOS)
  static img.Image _convertBGRA8888ToRGB(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    return img.Image.fromBytes(
      width: width,
      height: height,
      bytes: image.planes[0].bytes.buffer,
      order: img.ChannelOrder.bgra,
    );
  }

  static Future<Uint8List> convertYUV420ToJpeg(
    Uint8List yuvBytes, {
    required int width,
    required int height,
    required int yRowStride,
    required int uvRowStride,
    required int uvPixelStride,
    int? ySize,
    int? uSize,
    int? vSize,
    int quality = 90,
  }) async {
    final int actualYSize = ySize ?? yRowStride * height;
    final int actualUSize = uSize ?? uvRowStride * (height ~/ 2);
    final int actualVSize =
        vSize ?? (yuvBytes.length - actualYSize - actualUSize);

    final yPlane = yuvBytes.sublist(0, actualYSize);
    final uPlane = yuvBytes.sublist(actualYSize, actualYSize + actualUSize);
    final vPlane = yuvBytes.sublist(
        actualYSize + actualUSize, actualYSize + actualUSize + actualVSize);

    final rgbImage = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yRowStride + x;
        final int uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;

        final int yValue = yIndex < yPlane.length ? yPlane[yIndex] : 128;
        final int uValue = uvIndex < uPlane.length ? uPlane[uvIndex] : 128;
        final int vValue = uvIndex < vPlane.length ? vPlane[uvIndex] : 128;

        final int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
        final int g =
            (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
                .round()
                .clamp(0, 255);
        final int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);

        rgbImage.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    final jpegBytes = img.encodeJpg(rgbImage, quality: quality);
    return Uint8List.fromList(jpegBytes);
  }

  /// Convert raw YUV/BGRA planes to JPEG (for isolate worker)
  ///
  /// Used by IsolateImageProcessor to convert planes transferred from main isolate
  static Uint8List convertPlanesToJpeg(
    List<Uint8List> planes,
    int width,
    int height,
    int formatIndex,
    int quality,
    List<int> bytesPerRow,
    List<int> bytesPerPixel,
  ) {
    img.Image? rgbImage;

    // formatIndex corresponds to ImageFormatGroup enum index
    // CORRECT MAPPING:
    // 0 = unknown, 1 = yuv420, 2 = bgra8888, 3 = jpeg, 4 = nv21
    if (formatIndex == 1) {
      // YUV420 (ImageFormatGroup.yuv420)
      rgbImage = _convertYUV420FromPlanes(
        planes,
        width,
        height,
        bytesPerRow,
        bytesPerPixel,
      );
    } else if (formatIndex == 2) {
      // BGRA8888 (ImageFormatGroup.bgra8888)
      rgbImage = img.Image.fromBytes(
        width: width,
        height: height,
        bytes: planes[0].buffer,
        order: img.ChannelOrder.bgra,
      );
    } else {
      throw UnsupportedError(
          'Unsupported image format index: $formatIndex (expected 1=YUV420 or 2=BGRA8888)');
    }

    final jpegBytes = img.encodeJpg(rgbImage, quality: quality);
    return Uint8List.fromList(jpegBytes);
  }

  /// Convert YUV420 from raw planes (for isolate worker)
  static img.Image _convertYUV420FromPlanes(
    List<Uint8List> planes,
    int width,
    int height,
    List<int> bytesPerRow,
    List<int> bytesPerPixel,
  ) {
    final int yRowStride = bytesPerRow[0];
    final int uvRowStride = bytesPerRow[1];
    final int uvPixelStride = bytesPerPixel[1];

    // DEBUG: Log buffer details
    print('[YUV] ENTRY: width=$width, height=$height, yRowStride=$yRowStride');
    print(
        '[YUV] planes[0].length=${planes[0].length}, expected=${width * height}');

    // CRITICAL FIX: If yRowStride != width, we need to remove padding
    Uint8List yPlane = planes[0];
    Uint8List uPlane = planes[1];
    Uint8List vPlane = planes[2];

    if (yRowStride != width) {
      print(
          '[YUV] Removing padding from Y plane (stride=$yRowStride, width=$width)');
      // Create new buffer without padding
      final compactY = Uint8List(width * height);
      for (int y = 0; y < height; y++) {
        final srcOffset = y * yRowStride;
        final dstOffset = y * width;
        compactY.setRange(dstOffset, dstOffset + width, yPlane, srcOffset);
      }
      yPlane = compactY;
    }

    // For UV planes, check if stride != width/2
    final int uvWidth = width ~/ 2;
    final int uvHeight = height ~/ 2;

    if (uvRowStride != uvWidth) {
      print('[YUV] Removing padding from UV planes');
      final compactU = Uint8List(uvWidth * uvHeight);
      final compactV = Uint8List(uvWidth * uvHeight);

      for (int y = 0; y < uvHeight; y++) {
        for (int x = 0; x < uvWidth; x++) {
          final srcOffset = y * uvRowStride + x * uvPixelStride;
          final dstOffset = y * uvWidth + x;
          if (srcOffset < uPlane.length) {
            compactU[dstOffset] = uPlane[srcOffset];
          }
          if (srcOffset < vPlane.length) {
            compactV[dstOffset] = vPlane[srcOffset];
          }
        }
      }

      uPlane = compactU;
      vPlane = compactV;
    }

    final rgbImage = img.Image(width: width, height: height);

    // Now use compact buffers (padding removed)
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex =
            y * width + x; // Use width, not yRowStride (no padding now)
        final int uvIndex =
            (y ~/ 2) * uvWidth + (x ~/ 2); // Use uvWidth, not uvRowStride

        final int yValue = yPlane[yIndex];
        final int uValue = uPlane[uvIndex];
        final int vValue = vPlane[uvIndex];

        final int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
        final int g =
            (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
                .round()
                .clamp(0, 255);
        final int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);

        rgbImage.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return rgbImage;
  }
}
