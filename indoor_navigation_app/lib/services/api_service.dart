import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import '../config/app_config.dart';
import '../models/frame_data.dart';

class ApiService {
  // 스캔 시작
  static Future<String> startScan() async {
    final response = await http.post(
      Uri.parse('${AppConfig.baseUrl}${AppConfig.apiScanStart}'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'model': 'Unknown',
        'os': 'Android', // 또는 iOS
        'os_version': '14',
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['session_id'];
    } else {
      throw Exception('Failed to start scan: ${response.statusCode}');
    }
  }

  // 청크 업로드
  static Future<bool> uploadChunk({
    required String sessionId,
    required int chunkIndex,
    required List<FrameData> frames,
  }) async {
    final response = await http.post(
      Uri.parse('${AppConfig.baseUrl}${AppConfig.apiScanChunk}'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'session_id': sessionId,
        'chunk_index': chunkIndex,
        'frames': frames.map((f) => f.toJson()).toList(),
      }),
    );

    return response.statusCode == 200;
  }

  // 스캔 완료
  static Future<bool> finishScan(String sessionId) async {
    final response = await http.post(
      Uri.parse('${AppConfig.baseUrl}${AppConfig.apiScanFinish}'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'session_id': sessionId}),
    );

    return response.statusCode == 200;
  }

  // 상태 확인
  static Future<Map<String, dynamic>> getStatus(String sessionId) async {
    final response = await http.get(
      Uri.parse('${AppConfig.baseUrl}${AppConfig.apiScanStatus}/$sessionId'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to get status');
    }
  }

  // 맵 목록 조회
  static Future<List<Map<String, dynamic>>> getMaps() async {
    final response = await http.get(
      Uri.parse('${AppConfig.baseUrl}/api/maps'),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return List<Map<String, dynamic>>.from(data['maps']);
    } else {
      throw Exception('Failed to get maps: ${response.statusCode}');
    }
  }

  // 위치 추정
  static Future<Map<String, dynamic>> localize({
    required String mapId,
    required String imageBase64,
  }) async {
    final response = await http.post(
      Uri.parse('${AppConfig.baseUrl}${AppConfig.apiLocalize}'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'map_id': mapId,
        'image': imageBase64,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to localize');
    }
  }

  // 바이너리 청크 업로드 (base64 없음)
  static Future<bool> uploadChunkBinary({
    required String sessionId,
    required int chunkIndex,
    required List<FrameData> frames,
  }) async {
    try {
      final uri =
          Uri.parse('${AppConfig.baseUrl}${AppConfig.apiScanChunk}-binary');
      final request = http.MultipartRequest('POST', uri);

      // Add form fields
      request.fields['session_id'] = sessionId;
      request.fields['chunk_index'] = chunkIndex.toString();
      request.fields['num_frames'] = frames.length.toString();

      // Add binary JPEG files and depth files
      for (int i = 0; i < frames.length; i++) {
        final frame = frames[i];

        // RGB image
        request.files.add(http.MultipartFile.fromBytes(
          'images',
          frame.imageBytes,
          filename: 'frame_${chunkIndex}_$i.jpg',
          contentType: MediaType('image', 'jpeg'),
        ));

        // Depth map (if available)
        if (frame.depthBytes != null) {
          request.files.add(http.MultipartFile.fromBytes(
            'depths',
            frame.depthBytes!,
            filename: 'depth_${chunkIndex}_$i.raw',
            contentType: MediaType('application', 'octet-stream'),
          ));
        }
      }

      // Add metadata as JSON
      final metadata = {
        'timestamps': frames.map((f) => f.timestamp).toList(),
        'positions': frames.map((f) => f.position).toList(),
        'orientations': frames.map((f) => f.orientation).toList(),
        'imu_data': frames.map((f) => f.imuData).toList(),
        'camera_intrinsics': frames.first.cameraIntrinsics?.toJson(),
        'depth_widths': frames.map((f) => f.depthWidth).toList(),
        'depth_heights': frames.map((f) => f.depthHeight).toList(),
      };
      request.fields['metadata'] = jsonEncode(metadata);

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        print('✅ Binary chunk uploaded: $chunkIndex (${frames.length} frames)');
        return true;
      } else {
        print('❌ Upload failed: ${response.statusCode} - $responseBody');
        return false;
      }
    } catch (e) {
      print('❌ Upload error: $e');
      return false;
    }
  }
}
