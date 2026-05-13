from pydantic import BaseModel, ConfigDict, Field


class TrackingSessionStartRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "building_id": "e30f31ea-5bbe-42df-9031-fa371bb7a7b3",
                "min_confidence": 0.65,
                "min_matches": 180,
                "max_publish_age_ms": 5000,
            }
        }
    )

    building_id: str = Field(
        ...,
        description=(
            "Building/map id to localize against. Use the same id that is currently passed to "
            "`/api/slam/v3/localize` as `building_id`."
        ),
    )
    min_confidence: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum SuperPoint/PnP inlier-ratio confidence required before the server publishes "
            "a pose. This is not meter accuracy. Frames below this value still return "
            "`candidate_pose`, but `publish_pose` remains null."
        ),
    )
    min_matches: int = Field(
        180,
        ge=0,
        description=(
            "Minimum accepted PnP inlier match count. Raise this to make publishing stricter; "
            "lower it only when the app prefers more frequent but less reliable updates."
        ),
    )
    max_publish_age_ms: int = Field(
        5000,
        ge=0,
        description=(
            "Client freshness budget for `last_reliable_pose`. If the returned "
            "`last_reliable_age_ms` is larger than this, the client should treat the pose as stale "
            "and rely on local AR tracking/prediction until a new publish occurs."
        ),
    )


class TrackingSessionStartResponse(BaseModel):
    session_id: str = Field(..., description="In-memory tracking session id.")
    building_id: str = Field(..., description="Building/map id attached to the session.")
    status: str = Field(..., description="Session status.")
    thresholds: dict = Field(
        ...,
        description="Effective publish gates for this session: minConfidence, minMatches, maxPublishAgeMs.",
    )


class TrackingFrameResponse(BaseModel):
    session_id: str = Field(..., description="Tracking session id.")
    status: str = Field(
        ...,
        description=(
            "`localized`: this frame passed gates and `publish_pose` is new. "
            "`tracking`: this frame did not publish, but `last_reliable_pose` exists. "
            "`uncertain`: no reliable pose is currently available."
        ),
    )
    publish_pose: dict | None = Field(
        None,
        description=(
            "New reliable pose accepted from the submitted frame, in RTAB-Map/map coordinates. "
            "Use this to reset/update the client's map_T_AR transform. Null means this frame was "
            "not trusted enough to correct the client."
        ),
    )
    last_reliable_pose: dict | None = Field(
        None,
        description=(
            "Most recent published reliable pose in RTAB-Map/map coordinates. When `publish_pose` "
            "is null but this exists, the client may keep using local AR tracking from this anchor."
        ),
    )
    candidate_pose: dict | None = Field(
        None,
        description=(
            "Raw pose candidate from the current frame in RTAB-Map/map coordinates. This is for "
            "debug/telemetry only unless `quality.accepted=true`; do not directly snap the user to "
            "this value."
        ),
    )
    quality: dict = Field(
        ...,
        description=(
            "Frame quality and gate decisions. Important keys: `confidence`, `numMatches`, "
            "`accepted`, `rejectReasons`, `depthReceived`, `arPoseReceived`, `depthFusionApplied`, "
            "`depthFusion`, `depthCorrectionM`, `depthResidualMedianM`, `depthResidualMadM`, "
            "`depth3dPoseUsed`, `arPredictionApplied`, `arPriorConsistent`, and "
            "`arPriorPromoted`. During ongoing tracking, `resolvedPose`, `resolvedAnchor`, and "
            "`publishPoseSource` report whether the response used the continuously optimized "
            "full-6DoF map_T_AR anchor. If visual candidates disagree with the current AR prior "
            "but form a separate consistent cluster, `reanchorCandidateAccepted`, `reanchor`, and "
            "`reanchorApplied` explain whether a new map_T_AR anchor replaced the stale one. "
            "`outOfOrderLagMs`, `stateUpdateStale`, and `stateUpdateSkipped` show when a frame was "
            "processed after newer client timestamps and therefore could not overwrite session state. "
            "The server expects raw ARKit camera pose and converts ARKit camera axes to RTAB-Map "
            "`base_link` via OpenCV optical and the DB CameraModel optical rotation before computing `map_T_AR`. "
            "Query depth is used to solve metric 3D translation from "
            "matched feature depths. AR pose is used only as a session-relative prior/anchor; "
            "returned poses always remain in RTAB-Map/map coordinates."
        ),
    )
    frame_index: int = Field(..., description="0-based frame index in this session.")
    frame_timestamp_ms: int | None = Field(
        None,
        description=(
            "Client capture timestamp for the submitted frame. This should be the timestamp of the "
            "RGB/depth/AR pose sample in the client's monotonic or epoch millisecond clock."
        ),
    )
    server_received_ms: int = Field(
        ...,
        description="Server receive timestamp in epoch milliseconds. Use for latency/debug only.",
    )
    publish_timestamp_ms: int | None = Field(
        None,
        description=(
            "Client capture timestamp of `publish_pose`. When non-null, this tells the client "
            "which AR frame the map pose corresponds to."
        ),
    )
    last_reliable_timestamp_ms: int | None = Field(
        None,
        description=(
            "Client capture timestamp of `last_reliable_pose`. Use this to propagate the published "
            "map pose through subsequent ARKit deltas to the current render frame."
        ),
    )
    last_reliable_age_ms: int | None = Field(
        None,
        description="Age of lastReliablePose at response time.",
    )


class TrackingSessionStateResponse(BaseModel):
    session_id: str = Field(..., description="Tracking session id.")
    building_id: str = Field(..., description="Building/map id attached to the session.")
    status: str = Field(..., description="Session status.")
    frame_count: int = Field(..., description="Number of submitted frames.")
    last_reliable_pose: dict | None = Field(
        None,
        description="Most recent published reliable pose in RTAB-Map/map coordinates.",
    )
    last_candidate_pose: dict | None = Field(
        None,
        description="Most recent raw candidate pose. Debug only unless it was accepted.",
    )
    last_reliable_timestamp_ms: int | None = Field(
        None,
        description="Client capture timestamp of lastReliablePose.",
    )
    last_candidate_timestamp_ms: int | None = Field(
        None,
        description="Client capture timestamp of lastCandidatePose.",
    )
    last_reliable_age_ms: int | None = Field(None, description="Age of lastReliablePose.")
    thresholds: dict = Field(..., description="Quality gates used before publishing poses.")


class TrackingResolveRequest(BaseModel):
    timestamp_ms: int | None = Field(
        None,
        description="Client timestamp of the current AR pose to resolve into RTAB-Map/map coordinates.",
    )
    ar_pose: dict = Field(
        ...,
        description=(
            "Current client ARKit pose at `timestamp_ms`. Required shape: "
            "`{\"world_T_camera\":[16 row-major floats],\"trackingState\":\"normal\"}`. "
            "`world_T_camera` must be the same ARKit world coordinate convention sent to `/frames`; "
            "the server converts it to RTAB-Map/map coordinates using the accumulated map_T_AR "
            "anchor and never returns ARKit-local coordinates as the final pose. Send raw ARKit camera "
            "pose; the server applies ARKit-camera to RTAB-Map base_link axis conversion."
        ),
    )
    min_anchor_frames: int = Field(
        2,
        ge=1,
        description=(
            "Minimum number of consistent visual/depth anchor frames required before resolving. "
            "Use 2-3 for first localization; 1 is useful only for debugging."
        ),
    )
    max_anchor_age_ms: int | None = Field(
        30000,
        ge=0,
        description=(
            "Freshness window for the stored map_T_AR anchor sample selected by `timestamp_ms`. "
            "The server does not return a stored camera pose directly; it finds the nearest stored "
            "map_T_AR anchor sample and computes `map_T_base = map_T_AR * ar_pose.world_T_camera`. "
            "If the nearest sample is older/farther than this window, the response status becomes "
            "`tracking_only` while still returning the propagated pose for client-side continuity. "
            "Set null to disable this freshness downgrade."
        ),
    )


class TrackingResolveResponse(BaseModel):
    session_id: str = Field(..., description="Tracking session id.")
    status: str = Field(
        ...,
        description=(
            "`localized` when a fresh stored map_T_AR anchor sample exists, `tracking_only` when "
            "the nearest anchor is usable only for AR propagation because it is outside the requested "
            "freshness window, otherwise `initializing`."
        ),
    )
    pose: dict | None = Field(
        None,
        description="Resolved current pose in RTAB-Map/map coordinates at `pose_timestamp_ms`.",
    )
    pose_timestamp_ms: int | None = Field(
        None,
        description="Client timestamp of the AR pose used to compute `pose`.",
    )
    coordinate_convention: str = Field(
        "rtabmap_map",
        description="All returned poses use RTAB-Map/map coordinates.",
    )
    anchor: dict = Field(
        default_factory=dict,
        description=(
            "Robust map_T_AR anchor diagnostics: sourceFrames, inlierFrames, translationSpreadM, "
            "yawSpreadDeg, confidence, selected source frame indices, and rejection reason when "
            "status is `initializing`. `translationSpreadM` and `yawSpreadDeg` describe how tightly "
            "the accumulated visual/depth frames agree on the map_T_AR anchor. When localized, "
            "`optimizer` reports the full 6DoF SE(3) robust refinement status, linear/robust costs, "
            "translation residuals, and full rotation residuals."
            " For `/resolve`, `sampleTimestampMs`, `sampleTimestampDistanceMs`, and "
            "`requestedPoseTimestampMs` identify which stored map_T_AR anchor was used before "
            "multiplying the request AR pose into RTAB-Map/map coordinates."
        ),
    )
