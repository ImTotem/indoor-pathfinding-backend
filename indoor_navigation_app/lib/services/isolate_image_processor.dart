import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import '../utils/image_converter.dart';

/// Long-lived isolate for background image processing
class IsolateImageProcessor {
  Isolate? _isolate;
  SendPort? _sendPort;
  final _responsePort = ReceivePort();
  final _resultController = StreamController<Uint8List>.broadcast();

  /// Stream of processed JPEG bytes
  Stream<Uint8List> get processedFrames => _resultController.stream;

  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  /// Initialize the background worker isolate
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Use Completer to wait for handshake without closing port
      final handshakeCompleter = Completer<SendPort>();

      // Set up listener BEFORE spawning isolate (CRITICAL ORDER!)
      _responsePort.listen((message) {
        if (message is SendPort && !handshakeCompleter.isCompleted) {
          // First message: worker's SendPort (handshake)
          handshakeCompleter.complete(message);
        } else if (message is TransferableTypedData) {
          // Subsequent messages: processed JPEG bytes
          final jpegBytes = message.materialize().asUint8List();
          _resultController.add(jpegBytes);
        } else if (message is Map && message['error'] != null) {
          print('[IsolateImageProcessor] Error: ${message['error']}');
        }
      });

      // Spawn isolate AFTER listener is set up
      _isolate = await Isolate.spawn(
        _workerMain,
        _responsePort.sendPort,
        debugName: 'ImageProcessor',
      );

      // Wait for worker to send back its SendPort
      _sendPort = await handshakeCompleter.future;

      _isInitialized = true;
      print('[IsolateImageProcessor] Worker isolate initialized');
    } catch (e) {
      print('[IsolateImageProcessor] Initialization failed: $e');
      rethrow;
    }
  }

  /// Process a camera frame in the background isolate
  void processFrame(CameraImage image, int quality) {
    if (!_isInitialized || _sendPort == null) {
      print('[IsolateImageProcessor] Not initialized, dropping frame');
      return;
    }

    try {
      // Extract YUV planes from CameraImage
      final List<TransferableTypedData> planes = [];
      for (var plane in image.planes) {
        planes.add(TransferableTypedData.fromList([plane.bytes]));
      }

      // Send to worker with zero-copy transfer
      _sendPort!.send({
        'planes': planes,
        'width': image.width,
        'height': image.height,
        'format': image.format.group.index,
        'quality': quality,
        'bytesPerRow': image.planes.map((p) => p.bytesPerRow).toList(),
        'bytesPerPixel': image.planes.map((p) => p.bytesPerPixel ?? 1).toList(),
      });
    } catch (e) {
      print('[IsolateImageProcessor] Failed to send frame: $e');
    }
  }

  /// Dispose the worker isolate
  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
    _resultController.close();
    _responsePort.close();
    _isInitialized = false;
    print('[IsolateImageProcessor] Worker isolate disposed');
  }

  /// Worker isolate entry point
  static void _workerMain(SendPort mainSendPort) {
    final workerPort = ReceivePort();

    // Send worker's SendPort back to main isolate
    mainSendPort.send(workerPort.sendPort);

    // Listen for frames to process
    workerPort.listen((message) {
      try {
        if (message is! Map) return;

        final planes = (message['planes'] as List)
            .map(
                (t) => (t as TransferableTypedData).materialize().asUint8List())
            .toList();

        final width = message['width'] as int;
        final height = message['height'] as int;
        final formatIndex = message['format'] as int;
        final quality = message['quality'] as int;
        final bytesPerRow = (message['bytesPerRow'] as List).cast<int>();
        final bytesPerPixel = (message['bytesPerPixel'] as List).cast<int>();

        // Reconstruct CameraImage-like structure for conversion
        final jpegBytes = _convertToJpeg(
          planes,
          width,
          height,
          formatIndex,
          quality,
          bytesPerRow,
          bytesPerPixel,
        );

        // Send result back with zero-copy
        mainSendPort.send(TransferableTypedData.fromList([jpegBytes]));
      } catch (e, stackTrace) {
        print('[IsolateImageProcessor] Exception in worker: $e');
        print('[IsolateImageProcessor] Stack trace: $stackTrace');
        mainSendPort.send({'error': e.toString()});
      }
    });
  }

  /// Convert YUV/BGRA to JPEG in worker isolate
  static Uint8List _convertToJpeg(
    List<Uint8List> planes,
    int width,
    int height,
    int formatIndex,
    int quality,
    List<int> bytesPerRow,
    List<int> bytesPerPixel,
  ) {
    print(
        '[_convertToJpeg] Called with width=$width, height=$height, format=$formatIndex');
    print(
        '[_convertToJpeg] bytesPerRow=$bytesPerRow, planes[0].length=${planes[0].length}');

    // Import image package
    final img = ImageConverter.convertPlanesToJpeg(
      planes,
      width,
      height,
      formatIndex,
      quality,
      bytesPerRow,
      bytesPerPixel,
    );
    return img;
  }
}
