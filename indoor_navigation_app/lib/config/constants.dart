// lib/config/constants.dart
class AppConstants {
  // UI 상수
  static const double topBarOpacity = 0.7;
  static const double bottomBarOpacity = 0.8;
  static const double crosshairSize = 60.0;

  // 색상
  static const int colorGreen = 0xFF4CAF50;
  static const int colorRed = 0xFFF44336;
  static const int colorBlue = 0xFF2196F3;
  static const int colorOrange = 0xFFFF9800;
  static const int colorYellow = 0xFFFFEB3B;
  static const int colorCyan = 0xFF00BCD4;
  static const int colorPurple = 0xFF9C27B0;

  // 메시지
  static const String msgScanning = '천천히 주변을 둘러보세요 🔍';
  static const String msgReady = '평면을 탭하면 스캔 준비 완료 ✨';
  static const String msgStarted = '스캔 시작 (고품질 모드)';
  static const String msgPaused = '스캔 일시정지됨';
  static const String msgUploading = '데이터 전송 중...';

  // 움직임 가이드 메시지
  static const String msgMoveSlowly = '천천히 움직이세요 🐢';
  static const String msgMoveFaster = '조금 더 움직여주세요 →';
  static const String msgTooFast = '너무 빠릅니다! 천천히 🛑';
  static const String msgGoodMovement = '좋아요! 계속 움직이세요 ✨';
  static const String msgBlurry = '흔들림 감지 - 안정적으로 📷';
  static const String msgKeepSteady = '잠시 멈춰주세요 📸';

  // 에러 메시지
  static const String errStartFailed = '스캔 시작 실패';
  static const String errUploadFailed = '업로드 실패';
  static const String errFinishFailed = '완료 실패';
  static const String errCameraInit = '카메라 초기화 실패';

  // 위치 찾기 (Relocalization)
  static const String mapSelection = '지도 선택';
  static const String selectMapForRelocalization = '위치 찾기를 위한 지도를 선택하세요';
  static const String noMapsAvailable = '사용 가능한 지도가 없습니다';
  static const String keyframes = '키프레임';
  static const String loadingMaps = '지도 목록 불러오는 중...';
  static const String mapLoadError = '지도 목록을 불러올 수 없습니다';
  static const String retry = '다시 시도';
  static const String captureImages = '사진 촬영';
  static const String captureInstructions = '3장의 사진을 촬영하세요';
  static const String imagesCaptured = '장 촬영 완료';
  static const String tapToCapture = '탭하여 촬영';
  static const String localize = '위치 찾기';
  static const String localizing = '위치 인식 중...';
  static const String localizationResult = '위치 인식 결과';
  static const String position = '위치';
  static const String confidence = '신뢰도';
  static const String viewOn3DMap = '3D 지도에서 보기';
  static const String localizationFailed = '위치 인식 실패';
  static const String retryCapture = '다시 촬영';
}
