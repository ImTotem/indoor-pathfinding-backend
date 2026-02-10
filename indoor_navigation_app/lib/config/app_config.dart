class AppConfig {
  // 백엔드 서버 설정 (환경변수로 주입)
  // 사용법: flutter run --dart-define=BACKEND_HOST=192.168.1.100 --dart-define=BACKEND_PORT=8000
  static const String backendHost = String.fromEnvironment(
    'BACKEND_HOST',
    defaultValue: '100.89.34.75',
  );
  static const String backendPort = String.fromEnvironment(
    'BACKEND_PORT',
    defaultValue: '8000',
  );
  static String get baseUrl => 'http://$backendHost:$backendPort';

  // API 엔드포인트
  static const String apiScanStart = '/api/scan/start';
  static const String apiScanChunk = '/api/scan/chunk';
  static const String apiScanFinish = '/api/scan/finish';
  static const String apiScanStatus = '/api/scan/status';
  static const String apiLocalize = '/api/localize';
  static const String apiPathCalculate = '/api/path/calculate';

  // 스캔 설정
  static const int frameIntervalMs = 100; // 10 FPS 목표 (_isCapturing 가드가 과부하 방지)
  static const int chunkSize = 30; // 30 프레임마다 전송 (네트워크 효율)
  static const int maxStorageMB = 50; // 50MB 넘으면 강제 전송
  static const int imageQuality = 80; // JPEG 품질 (성능)
  static const int maxUploadQueueSize = 30; // 최대 업로드 큐 크기
  static const int imageWidth = 640;
  static const int imageHeight = 480;

  // 품질 검사 비활성화 (30 FPS 유지)
  static const bool enableQualityCheck = false;
  static const double minMovementDistance = 0.01; // 최소 이동 거리 (1cm)
  static const double minRotationAngle = 1.0; // 최소 회전 각도 (도)
  static const double maxMovementSpeed = 5.0; // 최대 이동 속도 (m/s)
  static const double blurThreshold = 10.0; // 블러 감지 임계값
  static const int minFramesBetweenKeyframes = 0; // 키프레임 간 최소 프레임 수
}
