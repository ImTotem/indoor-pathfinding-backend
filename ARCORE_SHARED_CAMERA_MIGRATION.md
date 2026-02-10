# ARCore Shared Camera 모드 전환 완료

## 변경 사항 요약

### 1. 새로운 통합 플러그인 생성 ✅
**파일**: `ARCoreSharedCameraPlugin.kt`
- ARCore Session을 `SHARED_CAMERA` 모드로 생성
- Camera2 API를 ARCore 래퍼로 감쌈
- RGB 프레임은 ImageReader로 캡처
- Depth는 `frame.acquireDepthImage16Bits()`로 획득
- 단일 카메라 세션으로 RGB + Depth 동시 캡처

### 2. Flutter 서비스 생성 ✅
**파일**: `arcore_shared_camera_service.dart`
- MethodChannel을 통한 네이티브 플러그인 호출
- `initialize()` → `startCamera()` → `captureFrame()` → `stopCamera()` → `dispose()` 라이프사이클

### 3. MainActivity 업데이트 ✅
- 기존 `ARCoreDepthPlugin` 대신 `ARCoreSharedCameraPlugin` 등록
- 채널명: `arcore_shared_camera`

## 주요 구현 특징

### ARCore Shared Camera 작동 방식
```kotlin
// 1. ARCore Session 생성 (Shared Camera 모드)
val session = Session(activity, EnumSet.of(Session.Feature.SHARED_CAMERA))
val sharedCamera = session.sharedCamera

// 2. Camera2 열기 (ARCore 래퍼 사용)
val wrappedCallback = sharedCamera.createARDeviceStateCallback(...)
cameraManager.openCamera(cameraId, wrappedCallback, handler)

// 3. Capture Session 생성 (모든 Surface 포함)
val surfaces = sharedCamera.arCoreSurfaces + rgbImageReader.surface
camera.createCaptureSession(surfaces, wrappedSessionCallback, handler)

// 4. 프레임 캡처
session.update()  // ARCore 업데이트
val depthImage = frame.acquireDepthImage16Bits()  // 깊이
val rgbImage = rgbImageReader.acquireLatestImage()  // RGB
```

### 카메라 충돌 해결 원리
- **기존 문제**: CameraX와 ARCore가 각각 독립적으로 카메라 열기 시도 → 충돌
- **해결 방법**: ARCore Shared Camera가 Camera2를 래핑하여 단일 세션 생성
  - ARCore가 메인 소유자
  - Camera2는 ARCore의 래퍼 콜백 사용
  - 모든 Surface(ARCore 내부 + 커스텀 ImageReader)를 하나의 CaptureSession에 등록

## 다음 단계: ScanningScreen 통합

ScanningScreen을 ARCoreSharedCameraService 사용하도록 수정해야 합니다.

### 변경 필요 사항

#### A. 서비스 초기화 (현재)
```dart
final CameraService _cameraService = CameraService();  // CameraX 기반
final ARCoreDepthService _depthService = ARCoreDepthService();  // 별도 ARCore
```

#### B. 서비스 초기화 (변경 후)
```dart
final ARCoreSharedCameraService _sharedCamera = ARCoreSharedCameraService();  // 통합
// CameraService, ARCoreDepthService 제거
```

#### C. 카메라 시작 로직 (현재)
```dart
// 별도 초기화
_depthInitialized = await _depthService.initDepth();
await _cameraService.startCapture();
```

#### D. 카메라 시작 로직 (변경 후)
```dart
// 통합 초기화
await _sharedCamera.initialize();
await _sharedCamera.startCamera();
```

#### E. 프레임 캡처 (현재)
```dart
// RGB: CameraX stream에서 폴링
final image = _cameraService.getLatestFrame();
// Depth: 별도 호출
final depthData = await _depthService.captureDepth();
```

#### F. 프레임 캡처 (변경 후)
```dart
// RGB + Depth 한 번에
final frameData = await _sharedCamera.captureFrame();
// frameData['rgbData'], frameData['depthData'] 포함
```

## 구현 진행 상황

- [x] ARCoreSharedCameraPlugin.kt 작성
- [x] ARCoreSharedCameraService.dart 작성  
- [x] MainActivity 플러그인 등록
- [ ] ScanningScreen 통합
- [ ] 프레임 처리 파이프라인 수정
- [ ] 빌드 및 테스트

## 예상 문제점 및 해결책

### 문제 1: YUV → JPEG 변환
ARCore SharedCamera는 YUV_420_888 포맷으로 프레임 제공

**해결**: 
- IsolateImageProcessor가 이미 YUV → JPEG 변환 지원
- `frameData['rgbData']`를 YUV로 처리하도록 수정

### 문제 2: 카메라 인트린직스
기존 CameraIntrinsicsPlugin은 별도 Camera2 세션 사용

**해결**:
- ARCore Session의 `cameraConfig`에서 인트린직스 추출
- 또는 기존 CameraIntrinsicsPlugin 유지 (한 번만 호출되므로 충돌 없음)

### 문제 3: ARCore Pose 데이터
ARCore는 카메라 포즈(위치/회전) 자동 제공

**현재**: 
```dart
position: [0.0, 0.0, 0.0],  // Identity (더미 값)
orientation: [0.0, 0.0, 0.0, 1.0]
```

**변경 후**:
```dart
position: frameData['position'],  // ARCore 실제 추정 값
orientation: frameData['orientation']
```

이는 **RTAB-Map의 초기 추정값으로 활용** 가능 (Visual Odometry 부트스트랩)

## 다음 작업

ScanningScreen 통합을 진행할까요? 아니면 먼저 빌드/테스트로 ARCoreSharedCameraPlugin이 동작하는지 확인할까요?

**추천**: 
1. 먼저 간단한 테스트 화면 만들어서 ARCoreSharedCameraPlugin 단독 테스트
2. 동작 확인 후 ScanningScreen 전체 통합

어느 쪽으로 진행할까요?
