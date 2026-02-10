import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:typed_data';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';

import '../services/camera_service.dart';
import '../config/app_config.dart';
import '../config/constants.dart';
import 'relocalization_result_screen.dart';

/// 재위치인식(Relocalization)을 위한 이미지 캡처 화면
class RelocalizationCaptureScreen extends StatefulWidget {
  final String mapId;

  const RelocalizationCaptureScreen({
    Key? key,
    required this.mapId,
  }) : super(key: key);

  @override
  State<RelocalizationCaptureScreen> createState() =>
      _RelocalizationCaptureScreenState();
}

class _RelocalizationCaptureScreenState
    extends State<RelocalizationCaptureScreen> {
  final CameraService _cameraService = CameraService();

  List<Uint8List> _capturedImages = [];
  bool _isUploading = false;
  bool _isCameraInitialized = false;
  int _selectedAlbumCount = 0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      await _cameraService.initialize();
      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      print('[Relocalization] 카메라 초기화 실패: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('카메라 초기화 실패: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _captureImage() async {
    if (!_isCameraInitialized || _cameraService.controller == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('카메라가 준비되지 않았습니다'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    try {
      final image = await _cameraService.controller!.takePicture();
      final imageBytes = await image.readAsBytes();

      setState(() {
        _capturedImages.add(imageBytes);
      });

      // 3번째 이미지 캡처 후 업로드
      if (_capturedImages.length == 3) {
        await _uploadImages();
      } else {
        // 피드백 표시
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('${_capturedImages.length}/3 촬영 완료'),
            backgroundColor: Color(AppConstants.colorGreen),
            duration: const Duration(milliseconds: 800),
          ),
        );
      }
    } catch (e) {
      print('[Relocalization] 이미지 캡처 실패: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('이미지 캡처 실패: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _selectFromAlbum() async {
    final ImagePicker picker = ImagePicker();

    try {
      // Pick 1-5 images with multi-select
      final List<XFile>? images = await picker.pickMultiImage();

      if (images == null || images.isEmpty) {
        // User cancelled picker
        return;
      }

      // Enforce 1-5 limit (Android may not honor limit param)
      final selectedImages = images.take(5).toList();

      setState(() {
        _selectedAlbumCount = selectedImages.length;
        _capturedImages.clear(); // Clear camera captures if any
      });

      // Convert XFile to bytes
      for (final xfile in selectedImages) {
        final bytes = await xfile.readAsBytes();
        _capturedImages.add(bytes);
      }

      // Upload immediately
      await _uploadImages();
    } catch (e) {
      print('[Relocalization] Album picker error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('앨범 선택 실패: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _uploadImages() async {
    if (_capturedImages.isEmpty || _capturedImages.length > 5) {
      return;
    }

    setState(() {
      _isUploading = true;
    });

    try {
      final uri = Uri.parse('${AppConfig.baseUrl}${AppConfig.apiLocalize}');
      final request = http.MultipartRequest('POST', uri);

      // Add map_id field
      request.fields['map_id'] = widget.mapId;

      // Add 3 image files
      for (int i = 0; i < _capturedImages.length; i++) {
        request.files.add(
          http.MultipartFile.fromBytes(
            'images',
            _capturedImages[i],
            filename: 'relocalization_$i.jpg',
            contentType: MediaType('image', 'jpeg'),
          ),
        );
      }

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        print('[Relocalization] 업로드 성공');

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('위치 인식 성공!'),
              backgroundColor: Colors.green,
            ),
          );

          // Navigate to result screen
          final poseData = jsonDecode(responseBody);
          Navigator.of(context).pushReplacement(
            MaterialPageRoute(
              builder: (context) =>
                  RelocationalizationResultScreen(poseData: poseData),
            ),
          );
        }
      } else {
        print(
            '[Relocalization] 업로드 실패: ${response.statusCode} - $responseBody');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('업로드 실패: ${response.statusCode}'),
              backgroundColor: Colors.red,
            ),
          );

          // Reset for retry
          setState(() {
            _capturedImages.clear();
          });
        }
      }
    } catch (e) {
      print('[Relocalization] 업로드 에러: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('업로드 에러: $e'),
            backgroundColor: Colors.red,
          ),
        );

        // Reset for retry
        setState(() {
          _capturedImages.clear();
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _cameraService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('이미지 캡처'),
        backgroundColor: Colors.blue,
      ),
      body: _isCameraInitialized && _cameraService.controller != null
          ? Stack(
              children: [
                // Camera preview
                SizedBox.expand(
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width:
                          _cameraService.controller!.value.previewSize!.height,
                      height:
                          _cameraService.controller!.value.previewSize!.width,
                      child: CameraPreview(_cameraService.controller!),
                    ),
                  ),
                ),

                // Counter overlay
                Positioned(
                  top: 20,
                  right: 20,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.6),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '${_capturedImages.length}/3',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),

                // Bottom controls
                Positioned(
                  bottom: 0,
                  left: 0,
                  right: 0,
                  child: SafeArea(
                    bottom: true,
                    child: Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.transparent,
                            Colors.black.withValues(alpha: 0.7),
                          ],
                        ),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 20,
                        ),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              _capturedImages.length < 3
                                  ? AppConstants.captureInstructions
                                  : AppConstants.localizing,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                              ),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 20),
                            if (!_isUploading)
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  ElevatedButton.icon(
                                    onPressed: _capturedImages.length < 3
                                        ? _captureImage
                                        : null,
                                    icon: const Icon(Icons.camera_alt),
                                    label: Text(
                                      _capturedImages.length < 3
                                          ? AppConstants.tapToCapture
                                          : '업로드 중...',
                                    ),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: Color(
                                        AppConstants.colorBlue,
                                      ),
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 24,
                                        vertical: 12,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  ElevatedButton.icon(
                                    onPressed: _selectFromAlbum,
                                    icon: const Icon(Icons.photo_library),
                                    label: Text(
                                      _selectedAlbumCount > 0
                                          ? '$_selectedAlbumCount개 선택됨'
                                          : '앨범에서 선택',
                                    ),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: Color(
                                        AppConstants.colorGreen,
                                      ),
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 24,
                                        vertical: 12,
                                      ),
                                    ),
                                  ),
                                ],
                              )
                            else
                              const SizedBox(
                                width: 50,
                                height: 50,
                                child: CircularProgressIndicator(
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    Colors.white,
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            )
          : Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  const Text('카메라 초기화 중...'),
                ],
              ),
            ),
    );
  }
}
