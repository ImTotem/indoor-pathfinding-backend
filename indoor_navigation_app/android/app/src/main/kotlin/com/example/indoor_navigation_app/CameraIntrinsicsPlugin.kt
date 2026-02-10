package com.example.indoor_navigation_app

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

class CameraIntrinsicsPlugin(private val context: Context) : MethodCallHandler {
    companion object {
        private const val CHANNEL = "camera_intrinsics"
        private const val TAG = "CameraIntrinsics"
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "getCameraIntrinsics" -> {
                try {
                    val intrinsics = getCameraIntrinsics()
                    result.success(intrinsics)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to get camera intrinsics", e)
                    result.error("INTRINSICS_ERROR", e.message, null)
                }
            }
            else -> result.notImplemented()
        }
    }

    private fun getCameraIntrinsics(): Map<String, Any> {
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        
        // Get back camera ID
        val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
            val characteristics = cameraManager.getCameraCharacteristics(id)
            characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        } ?: throw Exception("No back camera found")

        val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        // Extract intrinsic calibration (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
        val intrinsicCalibration = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            ?: throw Exception("LENS_INTRINSIC_CALIBRATION not available")

        // Extract distortion coefficients (k1, k2, p1, p2, k3, k4)
        val distortionCoeffs = characteristics.get(CameraCharacteristics.LENS_DISTORTION)
            ?: floatArrayOf(0f, 0f, 0f, 0f, 0f)  // Default to zero if not available

        // Get sensor size for image dimensions
        val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: throw Exception("Sensor size not available")

        // Parse intrinsic calibration:
        // [0]=fx, [1]=fy, [2]=cx, [3]=cy, [4]=s (skew, usually 0)
        val fx = intrinsicCalibration[0].toDouble()
        val fy = intrinsicCalibration[1].toDouble()
        val cx = intrinsicCalibration[2].toDouble()
        val cy = intrinsicCalibration[3].toDouble()

        // Parse distortion coefficients:
        // [0]=k1 (radial), [1]=k2 (radial), [2]=p1 (tangential), [3]=p2 (tangential), [4]=k3 (radial)
        val k1 = if (distortionCoeffs.size > 0) distortionCoeffs[0].toDouble() else 0.0
        val k2 = if (distortionCoeffs.size > 1) distortionCoeffs[1].toDouble() else 0.0
        val p1 = if (distortionCoeffs.size > 2) distortionCoeffs[2].toDouble() else 0.0
        val p2 = if (distortionCoeffs.size > 3) distortionCoeffs[3].toDouble() else 0.0

        val imageWidth = sensorSize.width()
        val imageHeight = sensorSize.height()

        Log.i(TAG, "Camera intrinsics: fx=$fx, fy=$fy, cx=$cx, cy=$cy, k1=$k1, k2=$k2, p1=$p1, p2=$p2, width=$imageWidth, height=$imageHeight")

        // Return 10 individual fields (NOT distortion array - see EXECUTE-WITH-CORRECTIONS.md)
        return mapOf(
            "fx" to fx,
            "fy" to fy,
            "cx" to cx,
            "cy" to cy,
            "k1" to k1,
            "k2" to k2,
            "p1" to p1,
            "p2" to p2,
            "width" to imageWidth,
            "height" to imageHeight
        )
    }
}
