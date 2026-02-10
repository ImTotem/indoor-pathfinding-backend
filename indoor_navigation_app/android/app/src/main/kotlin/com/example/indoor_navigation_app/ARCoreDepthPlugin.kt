package com.example.indoor_navigation_app

import android.app.Activity
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.util.Log
import com.google.ar.core.*
import com.google.ar.core.exceptions.*
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.nio.ByteBuffer

class ARCoreDepthPlugin(private val activity: Activity) : MethodCallHandler {
    companion object {
        private const val TAG = "ARCoreDepth"
    }

    private var session: Session? = null
    private var depthEnabled = false
    
    // EGL context for headless ARCore
    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglSurface: EGLSurface? = null
    private var glThread: Thread? = null
    private var hasSetTexture = false

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "initDepth" -> {
                try {
                    initializeARCoreDepth()
                    result.success(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to init ARCore depth", e)
                    result.error("INIT_ERROR", e.message, null)
                }
            }
            "captureDepth" -> {
                try {
                    val depthData = captureDepthMap()
                    if (depthData != null) {
                        result.success(depthData)
                    } else {
                        result.success(null)
                    }
                } catch (e: NotYetAvailableException) {
                    result.success(null)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to capture depth", e)
                    result.error("CAPTURE_ERROR", e.message, null)
                }
            }
            "disposeDepth" -> {
                try {
                    disposeARCoreSession()
                    result.success(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to dispose ARCore", e)
                    result.error("DISPOSE_ERROR", e.message, null)
                }
            }
            else -> result.notImplemented()
        }
    }

    private fun initializeARCoreDepth() {
        if (session != null) {
            Log.w(TAG, "ARCore session already initialized")
            return
        }

        createEGLContext()
        
        session = Session(activity)
        
        val config = session!!.config
        val isDepthSupported = session!!.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        
        if (!isDepthSupported) {
            Log.w(TAG, "Depth mode not supported on this device")
            throw UnsupportedConfigurationException("Depth not supported")
        }

        config.depthMode = Config.DepthMode.AUTOMATIC
        config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
        session!!.configure(config)
        
        session!!.resume()
        
        session!!.setCameraTextureNames(intArrayOf(0))
        hasSetTexture = true
        
        depthEnabled = true
        Log.i(TAG, "ARCore depth initialized with headless GL context")
    }
    
    private fun createEGLContext() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw RuntimeException("Unable to get EGL14 display")
        }
        
        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw RuntimeException("Unable to initialize EGL14")
        }
        
        val configAttribs = intArrayOf(
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
        if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)) {
            throw RuntimeException("Unable to find suitable EGL config")
        }
        
        val contextAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL14.EGL_NONE
        )
        
        eglContext = EGL14.eglCreateContext(
            eglDisplay, 
            configs[0], 
            EGL14.EGL_NO_CONTEXT, 
            contextAttribs, 
            0
        )
        
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            throw RuntimeException("Unable to create EGL context")
        }
        
        val surfaceAttribs = intArrayOf(
            EGL14.EGL_WIDTH, 1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE
        )
        
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0], surfaceAttribs, 0)
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw RuntimeException("Unable to create EGL surface")
        }
        
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw RuntimeException("Unable to make EGL context current")
        }
        
        Log.i(TAG, "EGL context created and bound successfully")
    }

    private fun captureDepthMap(): Map<String, Any>? {
        val currentSession = session ?: run {
            Log.w(TAG, "Session not initialized")
            return null
        }

        if (!depthEnabled) {
            Log.w(TAG, "Depth not enabled")
            return null
        }

        val frame = try {
            currentSession.update()
        } catch (e: CameraNotAvailableException) {
            Log.e(TAG, "Camera not available", e)
            return null
        }

        val depthImage = try {
            frame.acquireDepthImage16Bits()
        } catch (e: NotYetAvailableException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Failed to acquire depth image", e)
            return null
        }

        try {
            val width = depthImage.width
            val height = depthImage.height
            val timestamp = frame.timestamp

            val plane = depthImage.planes[0]
            val buffer: ByteBuffer = plane.buffer
            
            val depthBytes = ByteArray(buffer.remaining())
            buffer.get(depthBytes)

            Log.d(TAG, "Captured depth: ${width}x${height}, ${depthBytes.size} bytes, timestamp=$timestamp")

            return mapOf(
                "width" to width,
                "height" to height,
                "timestamp" to timestamp,
                "data" to depthBytes
            )
        } finally {
            depthImage.close()
        }
    }

    private fun disposeARCoreSession() {
        session?.pause()
        session?.close()
        session = null
        depthEnabled = false
        hasSetTexture = false
        
        eglSurface?.let { surface ->
            eglDisplay?.let { display ->
                EGL14.eglDestroySurface(display, surface)
            }
        }
        eglSurface = null
        
        eglContext?.let { context ->
            eglDisplay?.let { display ->
                EGL14.eglDestroyContext(display, context)
            }
        }
        eglContext = null
        
        eglDisplay?.let { display ->
            EGL14.eglTerminate(display)
        }
        eglDisplay = null
        
        Log.i(TAG, "ARCore session and EGL context disposed")
    }
}
