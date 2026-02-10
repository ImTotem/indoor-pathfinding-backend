import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../config/app_config.dart';
import '../config/constants.dart';

class RelocationalizationResultScreen extends StatelessWidget {
  final Map<String, dynamic> poseData;

  const RelocationalizationResultScreen({
    Key? key,
    required this.poseData,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppConstants.localizationResult),
        backgroundColor: const Color(AppConstants.colorBlue),
      ),
      body: _buildContent(context),
    );
  }

  Widget _buildContent(BuildContext context) {
    try {
      // Parse pose data
      final pose = poseData['pose'] as Map<String, dynamic>?;
      final confidence = poseData['confidence'] as num?;
      final mapId = poseData['map_id'] as String?;
      final numMatches = poseData['num_matches'] as int?;

      if (pose == null) {
        return _buildErrorWidget('위치 데이터가 없습니다');
      }

      final x = (pose['x'] as num?)?.toDouble() ?? 0.0;
      final y = (pose['y'] as num?)?.toDouble() ?? 0.0;
      final z = (pose['z'] as num?)?.toDouble() ?? 0.0;
      final confidencePercent = ((confidence ?? 0) * 100).toInt();

      return SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Success indicator
              Container(
                padding: const EdgeInsets.all(16.0),
                decoration: BoxDecoration(
                  color: const Color(AppConstants.colorGreen).withOpacity(0.1),
                  border: Border.all(
                    color: const Color(AppConstants.colorGreen),
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Row(
                  children: [
                    Icon(
                      Icons.check_circle,
                      color: Color(AppConstants.colorGreen),
                      size: 32,
                    ),
                    SizedBox(width: 12),
                    Text(
                      '위치 인식 성공',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(AppConstants.colorGreen),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Position section
              _buildSectionTitle('${AppConstants.position}'),
              const SizedBox(height: 12),
              _buildPositionCard(x, y, z),
              const SizedBox(height: 24),

              // Confidence section
              _buildSectionTitle('${AppConstants.confidence}'),
              const SizedBox(height: 12),
              _buildConfidenceCard(confidencePercent),
              const SizedBox(height: 24),

              // Additional info
              if (numMatches != null) ...[
                _buildSectionTitle('매칭 정보'),
                const SizedBox(height: 12),
                _buildInfoCard('매칭된 특징점', '$numMatches개'),
                const SizedBox(height: 24),
              ],

              if (mapId != null) ...[
                _buildSectionTitle('지도 정보'),
                const SizedBox(height: 12),
                _buildInfoCard('지도 ID', mapId),
                const SizedBox(height: 24),
              ],

              // View on 3D map button
              if (mapId != null)
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () => _openMapViewer(context, mapId, x, y, z),
                    icon: const Icon(Icons.map),
                    label: const Text(AppConstants.viewOn3DMap),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(AppConstants.colorBlue),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      textStyle: const TextStyle(fontSize: 16),
                    ),
                  ),
                ),
              const SizedBox(height: 16),

              // Back button
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  onPressed: () => Navigator.of(context).pop(),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    side: const BorderSide(
                      color: Color(AppConstants.colorBlue),
                      width: 2,
                    ),
                  ),
                  child: const Text(
                    '돌아가기',
                    style: TextStyle(
                      fontSize: 16,
                      color: Color(AppConstants.colorBlue),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    } catch (e) {
      return _buildErrorWidget('데이터 파싱 오류: $e');
    }
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.bold,
        color: Colors.black87,
      ),
    );
  }

  Widget _buildPositionCard(double x, double y, double z) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildPositionRow('X', x),
            const SizedBox(height: 12),
            _buildPositionRow('Y', y),
            const SizedBox(height: 12),
            _buildPositionRow('Z', z),
          ],
        ),
      ),
    );
  }

  Widget _buildPositionRow(String axis, double value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          '$axis 좌표',
          style: const TextStyle(
            fontSize: 14,
            color: Colors.black54,
          ),
        ),
        Text(
          '${value.toStringAsFixed(2)}m',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.black87,
          ),
        ),
      ],
    );
  }

  Widget _buildConfidenceCard(int confidencePercent) {
    final color = _getConfidenceColor(confidencePercent);

    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '신뢰도',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.black54,
                  ),
                ),
                Text(
                  '$confidencePercent%',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: confidencePercent / 100,
                minHeight: 8,
                backgroundColor: Colors.grey[300],
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoCard(String label, String value) {
    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 14,
                color: Colors.black54,
              ),
            ),
            Text(
              value,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorWidget(String message) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.error_outline,
              color: Color(AppConstants.colorRed),
              size: 64,
            ),
            const SizedBox(height: 16),
            Text(
              message,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 16,
                color: Colors.black87,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getConfidenceColor(int percent) {
    if (percent >= 80) {
      return const Color(AppConstants.colorGreen);
    } else if (percent >= 60) {
      return const Color(AppConstants.colorOrange);
    } else {
      return const Color(AppConstants.colorRed);
    }
  }

  Future<void> _openMapViewer(
    BuildContext context,
    String mapId,
    double x,
    double y,
    double z,
  ) async {
    try {
      final url =
          '${AppConfig.baseUrl}/api/viewer/map/$mapId?pose=${x.toStringAsFixed(2)},${y.toStringAsFixed(2)},${z.toStringAsFixed(2)}';

      if (await canLaunchUrl(Uri.parse(url))) {
        await launchUrl(
          Uri.parse(url),
          mode: LaunchMode.externalApplication,
        );
      } else {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('3D 지도를 열 수 없습니다'),
              backgroundColor: Color(AppConstants.colorRed),
            ),
          );
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('오류: $e'),
            backgroundColor: const Color(AppConstants.colorRed),
          ),
        );
      }
    }
  }
}
