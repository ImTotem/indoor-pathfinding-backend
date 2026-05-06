from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVER_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DB
    database_url: str = "postgresql+asyncpg://indoor:indoor@localhost:5432/indoor"

    # 인증
    ingest_api_token: str = "dev-token"

    # 파일 저장소
    storage_root: Path = _SERVER_ROOT / "var" / "storage"
    tmp_root: Path = _SERVER_ROOT / "var" / "tmp" / "uploads"

    # 업로드 상한 (bytes). 기본 2 GiB.
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024

    # 로그 레벨
    log_level: str = "INFO"

    # ── Sprint 16: Build Phase worker 설정 ────────────────────────────────────

    # 자동 빌드 트리거 (ingest 완료 시 build_job 자동 enqueue)
    build_auto_enqueue: bool = True

    # worker polling 주기 (초)
    build_worker_poll_interval_sec: float = 2.0

    # running job이 stale로 판단되는 경과 시간 (초)
    build_job_stale_after_sec: int = 600

    # 최대 재시도 횟수
    build_max_attempts: int = 3

    # 모델 캐시 디렉터리
    model_cache_dir: Path = _SERVER_ROOT / "var" / "models"

    # Segformer ONNX 모델 HF Hub 정보 (model_cache.py default와 일치)
    segformer_model_repo_id: str = "optimum/segformer-b0-finetuned-ade-512-512"
    segformer_model_filename: str = "model.onnx"

    # Walkable grid 셀 크기 (미터)
    walkable_grid_cell_m: float = 0.10

    # QualityGate 임계 — dev/시연 환경에서 데이터 품질이 낮을 때 완화 가능.
    # coverage = walkable cell 수 / grid bbox cell 수 (Sprint 22/63 — bbox 가
    # trajectory/RTABMap evidence 전체라 실 scan 에서도 0.30 미만이 흔함).
    # components = skeleton connected component 수.
    quality_gate_min_coverage: float = 0.10
    quality_gate_max_components: int = 3

    # rtabmap-reprocess multi-scan merge 의 graph optimizer max-error 임계 (m).
    # 두 chunk 의 ARKit world origin 차이가 큰 경우 (예: 40m) default 3.0 으로는
    # cross-session loop closure 가 outlier 로 reject 되어 pose 가 정렬되지 않음.
    multiscan_merge_optimize_max_error_m: float = 3.0

    # ── Sprint 17: 디버그 시각화 덤프 설정 ────────────────────────────────────

    # True 이면 워커가 빌드마다 var/debug/{scan_id}/{build_job_id}/ 에 덤프
    build_debug_dump: bool = False  # env: INDOOR_BUILD_DEBUG_DUMP

    # 디버그 덤프 루트 디렉터리
    debug_root: Path = _SERVER_ROOT / "var" / "debug"

    # ── Sprint 19: Depth Anything v2 NN 설정 ──────────────────────────────────

    # True 이면 BackProjectionStep 대신 DepthAwareBackProjectionStep 사용
    depth_nn_enabled: bool = False  # env: INDOOR_DEPTH_NN_ENABLED

    # 1차 HF Hub repo (onnx-community 공식 변환)
    depth_anything_model_repo_id: str = "onnx-community/depth-anything-v2-small"

    # 모델 파일명
    depth_anything_model_filename: str = "onnx/model.onnx"

    # fallback repo 후보 목록 (1차 실패 시 순서대로 시도)
    depth_anything_model_alternate_repos: list[str] = [
        "optimum/depth-anything-v2-small",
        "Xenova/depth-anything-v2-small",
    ]

    # ── Sprint 20: SuperPoint + LightGlue multi-view scale calibration ────────

    # True 이면 build pipeline에 MultiViewScaleCalibrationStep 삽입
    superpoint_lightglue_enabled: bool = False  # env: INDOOR_SUPERPOINT_LIGHTGLUE_ENABLED

    # 1차 HF Hub repo (fused SuperPoint+LightGlue 단일 ONNX)
    superpoint_lightglue_model_repo_id: str = "thomasonzhou/superpoint-lightglue"
    superpoint_lightglue_model_filename: str = "model.onnx"
    superpoint_lightglue_model_alternate_repos: list[str] = []

    # 각 이미지당 추출할 keypoint 상한 (모델 기본 1024)
    superpoint_max_keypoints: int = 1024

    # 인접 keyframe 창 크기 — keyframe i 당 (i+1..i+k) 와 pair 매칭
    superpoint_pair_window: int = 5

    # ONNX 입력 해상도 (grayscale, H=W). 작을수록 빠르고 정확도 낮아짐
    superpoint_input_size: int = 512

    # ── Dev Viewer 설정 ───────────────────────────────────────────────────────
    # True 이면 /dev/viewer/ 정적 파일 + /dev/viewer/scans 라우터 활성화
    dev_viewer_enabled: bool = False  # env: INDOOR_DEV_VIEWER_ENABLED

    # ── Sprint 37: RTAB-Map source-of-truth gate ─────────────────────────────

    # True이면 worker build 전에 uploaded rtabmap.db Node/Data/Feature readiness를
    # 검사하고, 부족하면 legacy keyframe/raw-pose path로 조용히 fallback하지 않는다.
    rtabmap_build_required: bool = True  # env: INDOOR_RTABMAP_BUILD_REQUIRED

    # ── Sprint 38: RTAB-Map trajectory road polygon ─────────────────────────

    # True이면 production worker가 keyframe jpg + raw pose back-projection 대신
    # rtabmap.db Node/Link trajectory에서 road polygon/grid를 만든다.
    # Sprint 46: floor_pointcloud_enabled로 source-of-truth가 이전됨에 따라 default off.
    rtabmap_trajectory_enabled: bool = False  # env: INDOOR_RTABMAP_TRAJECTORY_ENABLED

    # 지도식 복도 반폭. 실제 복도 폭이 아니라 Naver-like abstract road width.
    rtabmap_trajectory_half_width_m: float = 0.75

    # RTAB-Map Feature.depth_x/y/z cloud를 trajectory 주변 road width evidence로 사용.
    # Feature cloud는 침대/벽/가구 feature를 semantic 구분 없이 포함할 수 있으므로
    # floor-constrained filter가 생길 때까지 기본 경로에서는 끈다.
    rtabmap_feature_evidence_enabled: bool = False

    # RTAB-Map trajectory/feature road grid를 사각형 cover로 추상화해
    # 최종 walkable footprint를 무조건 직각 polygon으로 만든다.
    rtabmap_rectilinear_cover_enabled: bool = True
    rtabmap_rectilinear_cover_rotated_grid_enabled: bool = True

    # RTAB-Map Data.depth/calibration을 decode해 metric depth confidence grid를 만들고
    # rectilinear cover scoring에 약한 가중치로 연결한다.
    rtabmap_depth_evidence_enabled: bool = True
    rtabmap_depth_vertical_tolerance_m: float | None = 0.35

    # RTAB-Map Data.image를 Segformer로 segment해 floor/wall/stair mask를 만들고,
    # depth confidence/avoid projection에 사용한다. 모델 캐시/다운로드가 필요하므로 opt-in.
    rtabmap_image_segmentation_enabled: bool = False

    # Data.image가 sensor-native orientation으로 저장될 수 있으므로, Segformer 입력 전
    # orientation 후보를 선택한다. auto는 4방향 후보를 평가하고 mask를 원본 좌표계로 되돌린다.
    rtabmap_image_orientation_mode: str = "auto"

    # 너무 약한 floor mask와 과도한 wall mask는 depth confidence/avoid projection에서 제외한다.
    rtabmap_image_floor_mask_min_ratio: float = 0.02
    rtabmap_image_wall_mask_max_ratio: float = 0.55

    # ── Sprint 46: Floor segmentation point cloud as source-of-truth ─────────

    # True이면 production worker가 floor mask × per-pixel Data.depth ×
    # Data.calibration × Node.pose 를 back-project해 만든 world floor cloud를
    # walkable polygon의 source-of-truth로 사용한다. trajectory 기반 road
    # approximation을 대체하므로 침대/가구 위 trajectory가 polygon에 포함되는
    # 회귀를 차단한다.
    floor_pointcloud_enabled: bool = True  # env: INDOOR_FLOOR_POINTCLOUD_ENABLED

    # 픽셀 stride (4 → 1/16 down-sample). sparse floor mask 환경에서는 2로 낮춘다.
    floor_pointcloud_pixel_stride: int = 4

    # z0 ± tolerance 밖 점은 height filter에서 제거 (carrier/천장 노이즈 방어).
    floor_pointcloud_height_tolerance_m: float = 0.30

    # 10cm cell 당 최소 hit 수. 미만은 mask=False (sparse noise 방어).
    floor_pointcloud_min_cell_hits: int = 2

    # ── Sprint 47: CAD-style post-rectification polygon cleanup ──────────────
    # Sprint 47 codex-cross-review SELF_REVIEW_BIAS_DETECTED 권고에 따라 default OFF.
    # ON 전환은 Sprint 48에서 (a) dominant_angle_confidence>=0.55 path 마련 +
    # (b) 2개 scan 이상에서 corner_orthogonality_ratio>=0.90 / fallback_used=False
    # 통과 후. 현재 default ON은 raw_fallback cleanup으로 사용자에게 "CAD 적용됨"을
    # 잘못 신호하므로 OFF가 안전하다. opt-in: INDOOR_POLYGON_CAD_CLEANUP_ENABLED=true.
    polygon_cad_cleanup_enabled: bool = False

    # FloorRasterStep morph_close 적용 셀 수 — floor_pointcloud 경로에서만 명시 주입.
    # 전역 default 변경하지 않음 (W-5: 다른 호출자 영향 0).
    floor_raster_cad_morph_close_cells: int = 3

    # ManhattanRectification floor_pointcloud 모드 area_change 임계 (W-3 분리).
    # trajectory 등 default 0.20 영역은 그대로 유지된다.
    manhattan_floor_pointcloud_max_area_change: float = 0.55

    # PolygonCadCleanupStep 파라미터 (W-4 magic number config 노출).
    polygon_cad_collinear_angle_tol_deg: float = 5.0
    polygon_cad_short_edge_min_length_m: float = 0.20
    polygon_cad_near_vertex_merge_distance_m: float = 0.15
    polygon_cad_orthogonality_angle_tol_deg: float = 5.0

    # ── Sprint 48: dominant_angle hint chain ─────────────────────────────────
    # Codex critique F-3 권고대로 default ON 유지(hint helper 자체는 안전한 추가
    # 정보 — Manhattan rectification 호출자가 None 으로 보내면 회귀 0). 단
    # `polygon_cad_cleanup_enabled` 는 default False 그대로 유지(>=2 real scan
    # PASS 조건 미달성).
    dominant_angle_hint_enabled: bool = True
    # RTABMap link 후보 quality gate (Codex W-2)
    dominant_angle_hint_rtabmap_min_segments: int = 4
    dominant_angle_hint_rtabmap_min_total_length_m: float = 3.0
    # Sprint 49 hotfix: ㄴ자/T자 환경에서 두 축이 비슷한 weight라
    # length-weighted histogram의 best_bin이 0.18~0.30 수준에 머문다.
    # 0.45 gate는 직각 corridor (한 축만 dominant)에 맞춰진 값이라
    # multi-axis 환경에서 항상 reject 발화. 0.15로 완화하되 cross-check
    # 와 IOU shape preservation guard로 over-rectification 방어한다.
    dominant_angle_hint_rtabmap_min_best_bin_ratio: float = 0.15
    # footprint OBB 후보 gate (Codex W-3)
    # ㄴ자 polygon은 OBB가 거의 정사각형 (aspect ~1.1~1.5)이라 1.8 너무 엄격.
    dominant_angle_hint_obb_min_aspect_ratio: float = 1.3
    # cross-check (Codex W-10)
    dominant_angle_hint_cross_check_max_diff_deg: float = 15.0

    # ── Sprint 49: hint chain 4-way trigger (Codex Sprint 49 BLOCKER + W-2) ──
    # accepted=True 라도 다음 4 trigger 중 하나라도 발화하면 hint chain candidate
    # retry 를 수행한다 (Sprint 48 회귀 fix). orthogonality 임계는 config 노출.
    polygon_cad_hint_retry_orthogonality_threshold: float = 0.50
    # Codex W-5: pair_count_min_for_high_confidence (3 fallback 회피용 minimum) 와
    # acceptance pair_count (50 — scan_b real-scan 검증용) 분리.
    arkit_to_rtabmap_pair_count_min_for_high_confidence: int = 3
    arkit_to_rtabmap_residual_rms_max_m: float = 0.30

    # ── Sprint 50: rectangle dictionary cover (Codex BLOCKER 1~5) ─────────────
    # production default OFF. evidence PASS 후 별도 sync 단계에서 flip.
    # CLI/eval 경로 (run_real_scan --rectangle-cover) 에서만 ON.
    rectangle_dictionary_cover_enabled: bool = False
    rectangle_cover_precision_threshold: float = 0.85
    rectangle_cover_recall_min: float = 0.65
    rectangle_cover_over_cover_max: float = 0.20
    rectangle_cover_time_budget_sec: float = 30.0
    rectangle_cover_candidate_stride_cells: int = 3
    rectangle_cover_max_candidates_per_dimension: int = 200
    # Sprint 50 v2 — axis pair + dynamic precision threshold (production OFF
    # path 에서도 axes_mode/precision flag 노출, flip 시 그대로 사용 가능).
    rectangle_cover_axes_mode: str = "pair"
    rectangle_cover_precision_threshold_dynamic: bool = True
    rectangle_cover_precision_threshold_min: float = 0.65
    rectangle_cover_min_component_cells: int = 50
    rectangle_cover_axis_link_min_best_bin_ratio: float = 0.10
    rectangle_cover_axis_obb_min_aspect_ratio: float = 1.2

    # ── Sprint 51: Wall-fitting polygon (default OFF, observer only) ─────────
    # 7-step pipeline (obstacle heatmap → density → components → line fit →
    # snap → merge → assembly → validate). production polygon 무영향. 단
    # `use_floor_pointcloud=True` 와 함께 켜야 함 (floor mask + z0 의존).
    wall_polygon_enabled: bool = False  # env: INDOOR_WALL_POLYGON_ENABLED
    wall_polygon_density_min_cell_hits: int = 4
    wall_polygon_density_morph_close_radius_cells: int = 1
    wall_polygon_components_min_area_cells: int = 8
    wall_polygon_line_min_linearity: float = 0.85
    wall_polygon_line_min_length_m: float = 0.5
    wall_polygon_snap_tolerance_deg: float = 15.0
    wall_polygon_merge_offset_tolerance_m: float = 0.20
    wall_polygon_merge_gap_fill_m: float = 1.0
    wall_polygon_assembly_intersection_tolerance_m: float = 0.30
    wall_polygon_assembly_use_alpha_shape: bool = True
    wall_polygon_validate_floor_iou_min: float = 0.50
    wall_polygon_validate_area_change_max_ratio: float = 0.40
    wall_polygon_min_lines: int = 4
    wall_polygon_max_lines: int = 20
    wall_polygon_obstacle_height_min_m: float = 0.30
    wall_polygon_obstacle_height_max_m: float = 2.50

    # ── Sprint 61: 2D display navigation graph ──────────────────────────────
    # User-facing route graph. The real RTABMap graph/localization remains the
    # metric source of truth, then this graph provides a clean 2D map substrate.
    display_navigation_grid_enabled: bool = True
    display_navigation_grid_cell_m: float = 0.45
    display_navigation_grid_clearance_m: float = 0.30
    display_navigation_grid_connectivity: int = 8
    display_navigation_grid_poi_attach_k: int = 4

    # ── V1 VPS adapter boundary ─────────────────────────────────────────────
    # mock: CI/dev fallback. slam_v3: use legacy be `/api/slam/v3/localize`
    # implementation against v2 scan_ingest/floor_scan DB state.
    vps_localizer_mode: str = Field(
        default="mock",
        validation_alias=AliasChoices(
            "INDOOR_VPS_LOCALIZER_MODE", "VPS_LOCALIZER_MODE"
        ),
    )
    vps_shared_volume_root: Path | None = None
    vps_http_base_url: str | None = None

    # ── Sprint 78: mock passage seed ─────────────────────────────────────────
    # 개발/CI 환경에서 building 생성 시 1F/2F floor + elevator mock passage를 자동 seed.
    # production 배포 시 환경변수 INDOOR_SERVER_ENABLE_MOCK_PASSAGE=false 로 명시 OFF할 것.
    # AliasChoices: INDOOR_SERVER_ENABLE_MOCK_PASSAGE (권장) 또는 ENABLE_MOCK_PASSAGE (구버전 호환)
    enable_mock_passage: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "INDOOR_SERVER_ENABLE_MOCK_PASSAGE",
            "ENABLE_MOCK_PASSAGE",
        ),
    )


settings = Settings()
