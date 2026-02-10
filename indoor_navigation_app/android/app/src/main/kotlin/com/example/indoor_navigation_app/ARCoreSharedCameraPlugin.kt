package com.example.indoor_navigation_app

import android.app.Activity
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import com.google.ar.core.*
import com.google.ar.core.exceptions.*
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.EnumSet
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

// Three threads: Camera2 callbacks on backgroundHandler, session.update() on glHandler (EGL),
// Flutter calls on Main. captureFrame() bridges Main→GL via CountDownLatch.
class ARCoreSharedCameraPlugin(private val activity: Activity) : MethodCallHandler {
    companion object {
        private const val TAG = "ARCoreSharedCamera"
        private const val RGB_IMAGE_WIDTH = 1920
        private const val RGB_IMAGE_HEIGHT = 1080
        private const val RGB_MAX_IMAGES = 5
    }

    private var session: Session? = null
    private var sharedCamera: SharedCamera? = null
    private var cameraDevice: CameraDevice? = null
    private var cameraCaptureSession: CameraCaptureSession? = null
    private var rgbImageReader: ImageReader? = null

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private var glThread: HandlerThread? = null
    private var glHandler: Handler? = null

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var cameraTextureId: Int = -1

    private var isInitialized = false
    private var depthEnabled = false
    private val isResumed = AtomicBoolean(false)

    private var latestRgbImage: Image? = null
    private var camera2Intrinsics: FloatArray? = null

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "initialize" -> {
                try {
                    initialize()
                    result.success(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Initialization failed", e)
                    result.error("INIT_ERROR", e.message, null)
                }
            }
            "startCamera" -> {
                try {
                    startCamera()
                    result.success(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to start camera", e)
                    result.error("START_ERROR", e.message, null)
                }
            }
            "captureFrame" -> {
                try {
                    val frameData = captureFrame()
                    result.success(frameData)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to capture frame", e)
                    result.error("CAPTURE_ERROR", e.message, null)
                }
            }
            "stopCamera" -> {
                try {
                    stopCamera()
                    result.success(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to stop camera", e)
                    result.error("STOP_ERROR", e.message, null)
                }
            }
            "dispose" -> {
                try {
                    dispose()
                    result.success(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to dispose", e)
                    result.error("DISPOSE_ERROR", e.message, null)
                }
            }
            else -> result.notImplemented()
        }
    }

    private fun yuv420ToJpeg(image: Image, quality: Int): ByteArray {
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val uvHeight = height / 2

        val ySize = width * height
        val nv21 = ByteArray(ySize + width * uvHeight)

        val yBuf = yPlane.buffer
        yBuf.rewind()
        if (yRowStride == width) {
            yBuf.get(nv21, 0, ySize)
        } else {
            for (row in 0 until height) {
                yBuf.position(row * yRowStride)
                yBuf.get(nv21, row * width, width)
            }
        }

        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        uBuf.rewind()
        vBuf.rewind()

        if (uvPixelStride == 2) {
            // Semi-planar: V buffer contains interleaved VU (NV21) data
            for (row in 0 until uvHeight) {
                vBuf.position(row * uvRowStride)
                val toCopy = minOf(width, vBuf.remaining())
                vBuf.get(nv21, ySize + row * width, toCopy)
            }
        } else {
            var offset = ySize
            for (row in 0 until uvHeight) {
                for (col in 0 until width / 2) {
                    val idx = row * uvRowStride + col * uvPixelStride
                    if (idx < vBuf.limit()) nv21[offset] = vBuf.get(idx)
                    offset++
                    if (idx < uBuf.limit()) nv21[offset] = uBuf.get(idx)
                    offset++
                }
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, out)
        return out.toByteArray()
    }

    // Headless EGL context for ARCore's GL requirement (Grafika EglCore + ARCore #1375 pattern)
    private fun initEglContext() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw RuntimeException("Unable to get EGL14 display")
        }

        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw RuntimeException("Unable to initialize EGL14")
        }

        val attribList = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        if (!EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, 1, numConfigs, 0)) {
            throw RuntimeException("Unable to find suitable EGLConfig")
        }
        val eglConfig = configs[0] ?: throw RuntimeException("EGLConfig was null")

        val surfaceAttribs = intArrayOf(
            EGL14.EGL_WIDTH, 1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE
        )
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttribs, 0)
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw RuntimeException("Failed to create EGL PBuffer surface")
        }

        val contextAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL14.EGL_NONE
        )
        eglContext = EGL14.eglCreateContext(
            eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0
        )
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            throw RuntimeException("Failed to create EGL context")
        }

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw RuntimeException("eglMakeCurrent failed")
        }

        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        cameraTextureId = textures[0]

        Log.i(TAG, "EGL context created on GL thread, textureId=$cameraTextureId")
    }

    private fun releaseEglContext() {
        if (cameraTextureId >= 0) {
            GLES20.glDeleteTextures(1, intArrayOf(cameraTextureId), 0)
            cameraTextureId = -1
        }
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(
                eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT
            )
            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface)
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext)
            }
            EGL14.eglReleaseThread()
            EGL14.eglTerminate(eglDisplay)
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY
        eglContext = EGL14.EGL_NO_CONTEXT
        eglSurface = EGL14.EGL_NO_SURFACE
        Log.i(TAG, "EGL context released")
    }

    private fun initialize() {
        if (isInitialized) {
            Log.w(TAG, "Already initialized")
            return
        }

        startBackgroundThread()
        startGlThread()

        val eglLatch = CountDownLatch(1)
        var eglError: Exception? = null
        glHandler!!.post {
            try {
                initEglContext()
            } catch (e: Exception) {
                eglError = e
            } finally {
                eglLatch.countDown()
            }
        }
        if (!eglLatch.await(5, TimeUnit.SECONDS)) {
            throw RuntimeException("Timeout initializing EGL context")
        }
        eglError?.let { throw it }

        session = Session(activity, EnumSet.of(Session.Feature.SHARED_CAMERA))
        sharedCamera = session!!.sharedCamera

        val config = session!!.config
        val isDepthSupported = session!!.isDepthModeSupported(Config.DepthMode.AUTOMATIC)

        if (isDepthSupported) {
            config.depthMode = Config.DepthMode.AUTOMATIC
            depthEnabled = true
            Log.i(TAG, "Depth mode enabled")
        } else {
            config.depthMode = Config.DepthMode.DISABLED
            depthEnabled = false
            Log.w(TAG, "Depth mode not supported")
        }

        config.updateMode = Config.UpdateMode.BLOCKING
        session!!.configure(config)

        session!!.setCameraTextureName(cameraTextureId)
        Log.i(TAG, "setCameraTextureName($cameraTextureId) called")

        rgbImageReader = ImageReader.newInstance(
            RGB_IMAGE_WIDTH, RGB_IMAGE_HEIGHT,
            ImageFormat.YUV_420_888, RGB_MAX_IMAGES
        )

        rgbImageReader!!.setOnImageAvailableListener({ reader ->
            val newImage = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            synchronized(this) {
                latestRgbImage?.close()
                latestRgbImage = newImage
            }
        }, backgroundHandler)

        isInitialized = true
        Log.i(TAG, "ARCore Shared Camera initialized successfully")
    }

    private fun startCamera() {
        if (!isInitialized) {
            throw IllegalStateException("Not initialized")
        }

        val cameraManager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = session!!.cameraConfig.cameraId

        try {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val sensorIntrinsics = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val activeArray = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            if (sensorIntrinsics != null && sensorIntrinsics.size >= 4 && activeArray != null) {
                val scaleX = RGB_IMAGE_WIDTH.toFloat() / activeArray.width().toFloat()
                val scaleY = RGB_IMAGE_HEIGHT.toFloat() / activeArray.height().toFloat()
                camera2Intrinsics = floatArrayOf(
                    sensorIntrinsics[0] * scaleX,
                    sensorIntrinsics[1] * scaleY,
                    sensorIntrinsics[2] * scaleX,
                    sensorIntrinsics[3] * scaleY
                )
            } else {
                camera2Intrinsics = null
                Log.w(TAG, "Camera2 intrinsics unavailable for cameraId=$cameraId")
            }
        } catch (e: Exception) {
            camera2Intrinsics = null
            Log.w(TAG, "Failed to compute Camera2 intrinsics", e)
        }

        sharedCamera!!.setAppSurfaces(cameraId, listOf(rgbImageReader!!.surface))

        val wrappedCallback = sharedCamera!!.createARDeviceStateCallback(
            object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    Log.i(TAG, "Camera opened: $cameraId")
                    cameraDevice = camera
                    createCaptureSession()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    Log.w(TAG, "Camera disconnected")
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(TAG, "Camera error: $error")
                    camera.close()
                    cameraDevice = null
                }
            },
            backgroundHandler!!
        )

        try {
            cameraManager.openCamera(cameraId, wrappedCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to open camera", e)
            throw e
        }
    }

    private fun createCaptureSession() {
        val camera = cameraDevice ?: return

        val surfaces = sharedCamera!!.arCoreSurfaces.toMutableList()
        surfaces.add(rgbImageReader!!.surface)

        val wrappedSessionCallback = sharedCamera!!.createARSessionStateCallback(
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(captureSession: CameraCaptureSession) {
                    Log.i(TAG, "Capture session configured")
                    cameraCaptureSession = captureSession

                    try {
                        val captureRequest = camera.createCaptureRequest(
                            CameraDevice.TEMPLATE_RECORD
                        ).apply {
                            for (surface in surfaces) {
                                addTarget(surface)
                            }
                        }

                        captureSession.setRepeatingRequest(
                            captureRequest.build(),
                            object : CameraCaptureSession.CaptureCallback() {},
                            backgroundHandler
                        )

                        Log.i(TAG, "Camera streaming started, waiting for onActive...")
                    } catch (e: CameraAccessException) {
                        Log.e(TAG, "Failed to start repeating request", e)
                    }
                }

                // Resume in onActive (not onConfigured) per official SharedCameraActivity —
                // prevents SessionPausedException race condition
                override fun onActive(captureSession: CameraCaptureSession) {
                    Log.i(TAG, "Capture session active, resuming ARCore...")
                    try {
                        this@ARCoreSharedCameraPlugin.session?.resume()
                        isResumed.set(true)
                        Log.i(TAG, "ARCore session resumed successfully")
                    } catch (e: CameraNotAvailableException) {
                        Log.e(TAG, "Failed to resume ARCore session", e)
                    }
                }

                override fun onConfigureFailed(captureSession: CameraCaptureSession) {
                    Log.e(TAG, "Capture session configuration failed")
                }
            },
            backgroundHandler!!
        )

        try {
            camera.createCaptureSession(surfaces, wrappedSessionCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to create capture session", e)
        }
    }

    // session.update() must run on glHandler thread (EGL context bound there)
    private fun captureFrame(): Map<String, Any>? {
        if (session == null) {
            Log.w(TAG, "Session not initialized")
            return null
        }

        if (!isResumed.get()) {
            Log.w(TAG, "Session not yet resumed, skipping frame")
            return null
        }

        val latch = CountDownLatch(1)
        var frameResult: Map<String, Any>? = null
        var frameError: Exception? = null

        glHandler?.post {
            try {
                EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

                val frame = session!!.update()

                val result = mutableMapOf<String, Any>()

                synchronized(this@ARCoreSharedCameraPlugin) {
                    latestRgbImage?.let { rgbImage ->
                        try {
                            result["jpegData"] = yuv420ToJpeg(rgbImage, 80)
                            result["rgbWidth"] = rgbImage.width
                            result["rgbHeight"] = rgbImage.height
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed to convert image to JPEG", e)
                        }
                    }
                }

                try {
                    val cachedIntrinsics = camera2Intrinsics
                    if (cachedIntrinsics != null) {
                        result["focalLengthX"] = cachedIntrinsics[0].toDouble()
                        result["focalLengthY"] = cachedIntrinsics[1].toDouble()
                        result["principalPointX"] = cachedIntrinsics[2].toDouble()
                        result["principalPointY"] = cachedIntrinsics[3].toDouble()
                        result["imageWidth"] = RGB_IMAGE_WIDTH
                        result["imageHeight"] = RGB_IMAGE_HEIGHT
                    } else {
                        val arCamera = frame.camera
                        val intrinsics = arCamera.imageIntrinsics
                        val focalLength = intrinsics.focalLength
                        val principalPoint = intrinsics.principalPoint
                        val imageDimensions = intrinsics.imageDimensions
                        result["focalLengthX"] = focalLength[0].toDouble()
                        result["focalLengthY"] = focalLength[1].toDouble()
                        result["principalPointX"] = principalPoint[0].toDouble()
                        result["principalPointY"] = principalPoint[1].toDouble()
                        result["imageWidth"] = imageDimensions[0].toInt()
                        result["imageHeight"] = imageDimensions[1].toInt()
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to get camera intrinsics", e)
                }

                if (depthEnabled) {
                    try {
                        val depthImage = frame.acquireDepthImage16Bits()
                        val width = depthImage.width
                        val height = depthImage.height

                        val plane = depthImage.planes[0]
                        val buffer: ByteBuffer = plane.buffer
                        val depthBytes = ByteArray(buffer.remaining())
                        buffer.get(depthBytes)
                        depthImage.close()

                        result["depthData"] = depthBytes
                        result["depthWidth"] = width
                        result["depthHeight"] = height
                        result["depthTimestamp"] = frame.timestamp

                        Log.d(TAG, "Depth captured: ${width}x${height}")
                    } catch (_: NotYetAvailableException) {
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to acquire depth", e)
                    }
                }

                val camera = frame.camera
                result["timestamp"] = frame.timestamp
                result["trackingState"] = camera.trackingState.name

                val pose = camera.pose
                result["position"] = floatArrayOf(pose.tx(), pose.ty(), pose.tz())
                result["orientation"] = floatArrayOf(
                    pose.qx(), pose.qy(), pose.qz(), pose.qw()
                )

                // ARCore 특징점 추출 → 2D 스크린 좌표로 투영
                try {
                    val points = frame.getUpdatedTrackables(com.google.ar.core.Point::class.java)
                    if (points.isNotEmpty()) {
                        val arCamera = frame.camera
                        val viewMatrix = FloatArray(16)
                        val projMatrix = FloatArray(16)
                        arCamera.getViewMatrix(viewMatrix, 0)
                        arCamera.getProjectionMatrix(projMatrix, 0, 0.1f, 100.0f)

                        val screenPoints = mutableListOf<Float>()

                        for (point in points) {
                            if (point.trackingState != TrackingState.TRACKING) continue
                            val worldPos = point.pose
                            // World → Camera 변환
                            val wx = worldPos.tx(); val wy = worldPos.ty(); val wz = worldPos.tz()
                            val cx = viewMatrix[0]*wx + viewMatrix[4]*wy + viewMatrix[8]*wz + viewMatrix[12]
                            val cy = viewMatrix[1]*wx + viewMatrix[5]*wy + viewMatrix[9]*wz + viewMatrix[13]
                            val cz = viewMatrix[2]*wx + viewMatrix[6]*wy + viewMatrix[10]*wz + viewMatrix[14]

                            if (cz >= 0f) continue // 카메라 뒤쪽은 무시

                            // Camera → NDC
                            val ndcX = projMatrix[0]*cx + projMatrix[4]*cy + projMatrix[8]*cz + projMatrix[12]
                            val ndcY = projMatrix[1]*cx + projMatrix[5]*cy + projMatrix[9]*cz + projMatrix[13]
                            val ndcW = projMatrix[3]*cx + projMatrix[7]*cy + projMatrix[11]*cz + projMatrix[15]

                            if (ndcW == 0f) continue
                            val nx = ndcX / ndcW
                            val ny = ndcY / ndcW

                            // NDC → 이미지 좌표 (0~1 정규화)
                            val u = (nx + 1f) / 2f
                            val v = (1f - ny) / 2f

                            if (u in 0f..1f && v in 0f..1f) {
                                screenPoints.add(u)
                                screenPoints.add(v)
                            }
                        }
                        if (screenPoints.isNotEmpty()) {
                            result["featurePoints"] = screenPoints.toFloatArray()
                            result["featurePointCount"] = screenPoints.size / 2
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to get feature points", e)
                }

                frameResult = result

            } catch (e: SessionPausedException) {
                Log.w(TAG, "Session paused during update, skipping frame")
            } catch (e: MissingGlContextException) {
                Log.e(TAG, "GL context missing on GL thread!", e)
                frameError = e
            } catch (e: CameraNotAvailableException) {
                Log.e(TAG, "Camera not available", e)
            } catch (e: Exception) {
                Log.e(TAG, "Frame capture error", e)
                frameError = e
            } finally {
                latch.countDown()
            }
        }

        if (!latch.await(500, TimeUnit.MILLISECONDS)) {
            Log.w(TAG, "Timeout waiting for GL thread frame capture")
            return null
        }

        frameError?.let { throw it }
        return frameResult
    }

    private fun stopCamera() {
        isResumed.set(false)
        session?.pause()

        cameraCaptureSession?.close()
        cameraCaptureSession = null

        cameraDevice?.close()
        cameraDevice = null

        synchronized(this) {
            latestRgbImage?.close()
            latestRgbImage = null
        }

        Log.i(TAG, "Camera stopped")
    }

    private fun dispose() {
        stopCamera()

        rgbImageReader?.close()
        rgbImageReader = null

        val eglLatch = CountDownLatch(1)
        glHandler?.post {
            releaseEglContext()
            eglLatch.countDown()
        } ?: run { eglLatch.countDown() }
        eglLatch.await(2, TimeUnit.SECONDS)

        session?.close()
        session = null
        sharedCamera = null

        stopGlThread()
        stopBackgroundThread()

        isInitialized = false
        depthEnabled = false

        Log.i(TAG, "ARCore Shared Camera disposed")
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("ARCoreCamera2").apply {
            start()
            backgroundHandler = Handler(looper)
        }
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "Background thread interrupted", e)
        }
    }

    private fun startGlThread() {
        glThread = HandlerThread("ARCoreGL").apply {
            start()
            glHandler = Handler(looper)
        }
    }

    private fun stopGlThread() {
        glThread?.quitSafely()
        try {
            glThread?.join()
            glThread = null
            glHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "GL thread interrupted", e)
        }
    }
}
