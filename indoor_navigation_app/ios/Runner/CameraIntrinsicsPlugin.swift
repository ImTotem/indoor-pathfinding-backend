import Flutter
import UIKit
import AVFoundation

public class CameraIntrinsicsPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "camera_intrinsics", binaryMessenger: registrar.messenger())
        let instance = CameraIntrinsicsPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "getCameraIntrinsics":
            do {
                let intrinsics = try getCameraIntrinsics()
                result(intrinsics)
            } catch {
                result(FlutterError(code: "INTRINSICS_ERROR", message: error.localizedDescription, details: nil))
            }
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func getCameraIntrinsics() throws -> [String: Any] {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .back
        )
        
        guard let device = discoverySession.devices.first else {
            throw NSError(domain: "CameraIntrinsics", code: 1, userInfo: [NSLocalizedDescriptionKey: "No back camera found"])
        }
        
        let dimensions = device.activeFormat.formatDescription.dimensions
        let imageWidth = Int(dimensions.width)
        let imageHeight = Int(dimensions.height)
        
        let fx = Double(imageWidth) * 0.8
        let fy = Double(imageHeight) * 0.8
        let cx = Double(imageWidth) / 2.0
        let cy = Double(imageHeight) / 2.0
        
        print("[CameraIntrinsics] iOS intrinsics: fx=\(fx), fy=\(fy), cx=\(cx), cy=\(cy), k1=0.0, k2=0.0, p1=0.0, p2=0.0, width=\(imageWidth), height=\(imageHeight)")
        
        return [
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "width": imageWidth,
            "height": imageHeight
        ]
    }
}
