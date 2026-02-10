# ARCore SessionPausedException Fix

## Problem

```
E/ARCoreDepth: com.google.ar.core.exceptions.SessionPausedException
  at com.google.ar.core.Session.update(Session.java:2)
  
ARCoreError: Cannot update frame, session is paused
AR_ERROR_SESSION_PAUSED
```

## Root Cause

ARCore sessions are created in a **PAUSED** state by default. The `session.update()` method cannot be called on a paused session.

### What Was Missing

```kotlin
// WRONG - session is paused after configure()
session = Session(activity)
session!!.configure(config)
// ❌ No resume() call
val frame = session.update()  // ← SessionPausedException!
```

## Solution

Call `session.resume()` after configuration and before any `update()` calls:

```kotlin
// CORRECT - explicitly resume session
session = Session(activity)
session!!.configure(config)
session!!.resume()  // ✅ Resume before update()
val frame = session.update()  // Works!
```

## Changes Made

### 1. Add `resume()` in `initializeARCoreDepth()`

**File**: `ARCoreDepthPlugin.kt` line 80

```kotlin
config.depthMode = Config.DepthMode.AUTOMATIC
config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
session!!.configure(config)

session!!.resume()  // ← Added

depthEnabled = true
Log.i(TAG, "ARCore depth initialized and resumed successfully")
```

### 2. Add `pause()` in `disposeARCoreSession()`

**File**: `ARCoreDepthPlugin.kt` line 138

```kotlin
private fun disposeARCoreSession() {
    session?.pause()  // ← Added for clean shutdown
    session?.close()
    session = null
    depthEnabled = false
}
```

## ARCore Session Lifecycle

```
CREATE → CONFIGURE → RESUME → UPDATE (loop) → PAUSE → CLOSE
         ^                ^                    ^
         |                |                    |
    Config.DepthMode   Required!         Clean disposal
```

### Key Rules

1. **After `Session()` creation**: Session is PAUSED
2. **After `configure()`**: Session is still PAUSED
3. **`resume()` must be called**: Before any `update()` calls
4. **`pause()` should be called**: Before `close()` for clean shutdown

## Evidence from ARCore Samples

All official ARCore samples follow this pattern:

```java
// Google ARCore samples (HelloAR, DrawAR, etc.)
mSession = new Session(activity);
mSession.configure(config);
mSession.resume();  // ← Always present
```

### Real Examples

1. **google/ar-drawing-java**: [DrawAR.java#L330](https://github.com/googlecreativelab/ar-drawing-java/blob/master/app/src/main/java/com/googlecreativelab/drawar/DrawAR.java#L330)
2. **google/justaline-android**: [DrawARActivity.java#L377](https://github.com/googlecreativelab/justaline-android/blob/master/app/src/main/java/com/arexperiments/justaline/DrawARActivity.java#L377)
3. **aws-samples/sumerian-arcore**: [MainActivity.java#L154](https://github.com/aws-samples/amazon-sumerian-arcore-starter-app/blob/master/SumerianARCoreStarter/app/src/main/java/com/amazon/sumerianarcorestarter/MainActivity.java#L154)

All include the pattern:
```java
mSession.configure(config);
mSession.resume();  // ← Required
```

## Testing

After this fix:
1. ✅ `initDepth()` should succeed without errors
2. ✅ `captureDepth()` should return depth data (no SessionPausedException)
3. ✅ Logs should show: "ARCore depth initialized and resumed successfully"

## References

- ARCore Official Docs: [Session Lifecycle](https://developers.google.com/ar/develop/java/enable-arcore#session-lifecycle)
- Error Space: `ArStatusErrorSpace::AR_ERROR_SESSION_PAUSED` indicates session not resumed
