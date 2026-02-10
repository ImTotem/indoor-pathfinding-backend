// lib/screens/processing_screen.dart
import 'package:flutter/material.dart';
import 'dart:async';
import '../services/api_service.dart';
import 'map_viewer_screen.dart';

class ProcessingScreen extends StatefulWidget {
  final String sessionId;
  
  ProcessingScreen({required this.sessionId});
  
  @override
  _ProcessingScreenState createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  double progress = 0.0;
  String status = 'processing';
  Timer? _pollTimer;
  
  @override
  void initState() {
    super.initState();
    _startPolling();
  }
  
  void _startPolling() {
    // 2초마다 상태 확인
    _pollTimer = Timer.periodic(Duration(seconds: 2), (timer) async {
      try {
        final statusData = await ApiService.getStatus(widget.sessionId);
        
        setState(() {
          status = statusData['status'];
          progress = (statusData['progress'] ?? 0).toDouble();
        });
        
        if (status == 'completed') {
          timer.cancel();
          _showCompletedDialog(statusData['map_id']);
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => MapViewerScreen(mapId: statusData['map_id']!),
            ),
          );
        } else if (status == 'failed') {
          timer.cancel();
          _showErrorDialog(statusData['error']);
        }
      } catch (e) {
        print('상태 확인 실패: $e');
      }
    });
  }
  
  void _showCompletedDialog(String mapId) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('맵 생성 완료!'),
        content: Text('맵 ID: $mapId'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context); // 홈으로
            },
            child: Text('확인'),
          ),
        ],
      ),
    );
  }
  
  void _showErrorDialog(String? error) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('처리 실패'),
        content: Text(error ?? '알 수 없는 에러'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context);
            },
            child: Text('확인'),
          ),
        ],
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('맵 생성 중'),
        automaticallyImplyLeading: false,
      ),
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(40),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(value: progress / 100),
              SizedBox(height: 40),
              Text(
                '${progress.toInt()}%',
                style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 20),
              Text(
                'SLAM 처리 중입니다...',
                style: TextStyle(fontSize: 16, color: Colors.grey),
              ),
              SizedBox(height: 10),
              Text(
                '세션: ${widget.sessionId}',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }
}

