package com.example.indoor_navigation_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CAMERA_INTRINSICS_CHANNEL = "camera_intrinsics"
    private val ARCORE_SHARED_CAMERA_CHANNEL = "arcore_shared_camera"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CAMERA_INTRINSICS_CHANNEL).setMethodCallHandler(
            CameraIntrinsicsPlugin(this)
        )
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, ARCORE_SHARED_CAMERA_CHANNEL).setMethodCallHandler(
            ARCoreSharedCameraPlugin(this)
        )
    }
}
