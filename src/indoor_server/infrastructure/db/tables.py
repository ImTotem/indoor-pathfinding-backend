"""SQLAlchemy Core Table 정의. Alembic이 이 메타데이터를 참조한다."""
from __future__ import annotations

import sqlalchemy as sa
from geoalchemy2 import Geometry
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID

metadata = sa.MetaData()

# ── Enum 타입 ─────────────────────────────────────────────────────────────────

poi_source_enum = sa.Enum("track_lock", "manual", name="poi_source")
detection_source_enum = sa.Enum("on_device", "server", name="detection_source")
# Sprint 16: queued → pending, built → succeeded, cancelled 추가
build_state_enum = sa.Enum(
    "not_started", "pending", "running", "succeeded", "failed", "cancelled",
    name="build_state",
)
node_type_enum = sa.Enum(
    "junction", "endpoint", "corridor", "poi", "poi_attach",
    "passage_stairs", "passage_elevator", "passage_escalator",
    name="node_type",
)
edge_type_enum = sa.Enum("skeleton", "poi_spur", name="edge_type")
build_step_enum = sa.Enum(
    "init", "floor_seg", "back_project", "walkable_grid", "skeleton",
    "node_placement", "poi_projection", "quality_gate", "persist", "done",
    name="build_step",
)
build_failure_reason_enum = sa.Enum(
    "walkable_coverage_low", "graph_disconnected", "model_load_failed",
    "intrinsics_missing", "rtabmap_data_not_ready", "internal",
    name="build_failure_reason",
)

# ── V1 building/floor compatibility ──────────────────────────────────────────

building = sa.Table(
    "building",
    metadata,
    sa.Column(
        "building_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("description", sa.Text, nullable=True),
    sa.Column("latitude", sa.Float, nullable=True),
    sa.Column("longitude", sa.Float, nullable=True),
    sa.Column("status", sa.Text, nullable=False, server_default="DRAFT"),
    sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.Column(
        "updated_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.CheckConstraint(
        "status IN ('DRAFT','ACTIVE')",
        name="ck_building_status",
    ),
)

building_floor = sa.Table(
    "building_floor",
    metadata,
    sa.Column(
        "floor_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column(
        "building_id",
        UUID(as_uuid=False),
        sa.ForeignKey("building.building_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("level", sa.Integer, nullable=False),
    sa.Column("height", sa.Float, nullable=True),
    sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.Column(
        "updated_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.UniqueConstraint("building_id", "level", name="uq_building_floor_level"),
)

floor_scan = sa.Table(
    "floor_scan",
    metadata,
    sa.Column(
        "floor_scan_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column(
        "floor_id",
        UUID(as_uuid=False),
        sa.ForeignKey("building_floor.floor_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("file_name", sa.Text, nullable=True),
    sa.Column("file_size", sa.BigInteger, nullable=True),
    sa.Column("status", sa.Text, nullable=False, server_default="UPLOADED"),
    sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
    sa.Column("upload_order", sa.Integer, nullable=False, server_default="0"),
    sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.UniqueConstraint("floor_id", "scan_id", name="uq_floor_scan_scan"),
    sa.Index("ix_floor_scan_floor", "floor_id"),
    sa.Index("ix_floor_scan_scan", "scan_id"),
)

# ── scan_ingest ───────────────────────────────────────────────────────────────

scan_ingest = sa.Table(
    "scan_ingest",
    metadata,
    sa.Column("scan_id", UUID(as_uuid=False), primary_key=True),
    sa.Column("payload_sha256", sa.Text, nullable=False),
    sa.Column(
        "ingested_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.Column("replaced_at", sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column("storage_path", sa.Text, nullable=False),
    sa.Column("device_info", JSONB, nullable=True),
    sa.Column(
        "build_state",
        build_state_enum,
        nullable=False,
        server_default="not_started",
    ),
    sa.Column("build_job_id", UUID(as_uuid=False), nullable=True),
)

# ── scan_session ──────────────────────────────────────────────────────────────

scan_session = sa.Table(
    "scan_session",
    metadata,
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("started_at", sa.BigInteger, nullable=False),
    sa.Column("ended_at", sa.BigInteger, nullable=True),
    sa.Column("device_model", sa.Text, nullable=False),
    sa.Column("app_version", sa.Text, nullable=False),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("keyframe_count", sa.Integer, nullable=False, server_default="0"),
    sa.Column("notes", sa.Text, nullable=True),
)

# ── keyframe_meta ─────────────────────────────────────────────────────────────

keyframe_meta = sa.Table(
    "keyframe_meta",
    metadata,
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_session.scan_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("seq", sa.Integer, nullable=False),
    sa.Column("captured_at", sa.BigInteger, nullable=False),
    sa.Column("image_path", sa.Text, nullable=False),
    sa.Column("pose_matrix", BYTEA, nullable=False),
    sa.Column("tx", sa.Float, nullable=False),
    sa.Column("ty", sa.Float, nullable=False),
    sa.Column("tz", sa.Float, nullable=False),
    sa.Column("tracking_state", sa.Text, nullable=False),
    sa.Column("rtabmap_node_id", sa.Integer, nullable=True),
    sa.PrimaryKeyConstraint("scan_id", "seq"),
    sa.Index("ix_keyframe_meta_scan_captured", "scan_id", "captured_at"),
)

# ── poi_canonical (후속 스프린트 R-4 placeholder) ──────────────────────────────

poi_canonical = sa.Table(
    "poi_canonical",
    metadata,
    sa.Column(
        "canonical_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column("label", sa.Text, nullable=True),
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
        nullable=True,
    ),
    sa.Column(
        "building_id",
        UUID(as_uuid=False),
        sa.ForeignKey("building.building_id", ondelete="CASCADE"),
        nullable=True,
    ),
    sa.Column(
        "floor_id",
        UUID(as_uuid=False),
        sa.ForeignKey("building_floor.floor_id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.Column("level_id", sa.Text, nullable=False, server_default="level-0"),
    sa.Column("category", sa.Text, nullable=False, server_default="unknown"),
    sa.Column("name", sa.Text, nullable=True),
    sa.Column("world_pose", Geometry("POINTZ", srid=0), nullable=True),
    sa.Column(
        "route_node_id",
        UUID(as_uuid=False),
        sa.ForeignKey("map_node.node_id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.Column("display_point", Geometry("POINTZ", srid=0), nullable=True),
    sa.Column("display_area_id", UUID(as_uuid=False), nullable=True),
    sa.Column("source_mark_ids", JSONB, nullable=True),
    sa.Column("needs_review", sa.Boolean, nullable=False, server_default="false"),
    sa.Column("cluster_method", sa.Text, nullable=True),
    sa.Column("llm_confidence", sa.Float, nullable=True),
    sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
)

# ── place_area / semantic labels / vertical connectors ───────────────────────

place_area = sa.Table(
    "place_area",
    metadata,
    sa.Column(
        "place_area_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("level_id", sa.Text, nullable=False, server_default="level-0"),
    sa.Column("category", sa.Text, nullable=False),
    sa.Column("name", sa.Text, nullable=True),
    sa.Column("geom", Geometry("MULTIPOLYGON", srid=0), nullable=False),
    sa.Column(
        "entrance_node_id",
        UUID(as_uuid=False),
        sa.ForeignKey("map_node.node_id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.Column(
        "source_poi_id",
        sa.BigInteger,
        sa.ForeignKey("poi_mark.id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.Column("source", sa.Text, nullable=False, server_default="mock"),
    sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
)

poi_analysis_job = sa.Table(
    "poi_analysis_job",
    metadata,
    sa.Column(
        "analysis_job_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column(
        "poi_mark_id",
        sa.BigInteger,
        sa.ForeignKey("poi_mark.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("analyzer", sa.Text, nullable=False),
    sa.Column("state", sa.Text, nullable=False, server_default="succeeded"),
    sa.Column("result_json", JSONB, nullable=True),
    sa.Column("error", sa.Text, nullable=True),
    sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
)

place_label_candidate = sa.Table(
    "place_label_candidate",
    metadata,
    sa.Column(
        "candidate_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column(
        "canonical_id",
        UUID(as_uuid=False),
        sa.ForeignKey("poi_canonical.canonical_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("source", sa.Text, nullable=False),
    sa.Column("label", sa.Text, nullable=True),
    sa.Column("category", sa.Text, nullable=True),
    sa.Column("confidence", sa.Float, nullable=False, server_default="0"),
    sa.Column("raw_json", JSONB, nullable=True),
)

vertical_connector = sa.Table(
    "vertical_connector",
    metadata,
    sa.Column(
        "connector_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column(
        "building_id",
        UUID(as_uuid=False),
        sa.ForeignKey("building.building_id", ondelete="CASCADE"),
        nullable=True,
    ),
    sa.Column("connector_type", sa.Text, nullable=False),
    sa.Column("connector_key", sa.Text, nullable=False),
    sa.Column("name", sa.Text, nullable=True),
    sa.Column(
        "is_mock",
        sa.Boolean,
        nullable=False,
        server_default=sa.text("false"),
    ),
    sa.Column(
        "created_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.UniqueConstraint("building_id", "connector_type", "connector_key"),
)

vertical_connector_stop = sa.Table(
    "vertical_connector_stop",
    metadata,
    sa.Column(
        "connector_stop_id",
        UUID(as_uuid=False),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    ),
    sa.Column(
        "connector_id",
        UUID(as_uuid=False),
        sa.ForeignKey("vertical_connector.connector_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("level_id", sa.Text, nullable=False),
    sa.Column(
        "poi_canonical_id",
        UUID(as_uuid=False),
        sa.ForeignKey("poi_canonical.canonical_id", ondelete="CASCADE"),
        nullable=True,
    ),
    sa.Column(
        "route_node_id",
        UUID(as_uuid=False),
        sa.ForeignKey("map_node.node_id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.UniqueConstraint("connector_id", "level_id"),
)

# ── poi_mark ──────────────────────────────────────────────────────────────────

poi_mark = sa.Table(
    "poi_mark",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
    sa.Column("keyframe_seq", sa.Integer, nullable=False),
    sa.Column("created_at", sa.BigInteger, nullable=False),
    sa.Column("pose_matrix", BYTEA, nullable=False),
    sa.Column("tx", sa.Float, nullable=False),
    sa.Column("ty", sa.Float, nullable=False),
    sa.Column("tz", sa.Float, nullable=False),
    sa.Column("track_id", sa.Integer, nullable=True),
    sa.Column("label", sa.Text, nullable=True),
    sa.Column("source", poi_source_enum, nullable=False),
    # R-2 triangulation 결과 수용 placeholder — 본 스프린트 NULL
    sa.Column("world_pose", Geometry("POINTZ", srid=0), nullable=True),
    # R-4 POI canonical merge placeholder — 본 스프린트 NULL
    sa.Column(
        "canonical_id",
        UUID(as_uuid=False),
        sa.ForeignKey("poi_canonical.canonical_id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.ForeignKeyConstraint(
        ["scan_id", "keyframe_seq"],
        ["keyframe_meta.scan_id", "keyframe_meta.seq"],
        ondelete="CASCADE",
    ),
)

# partial unique index: iOS v3 정책 승계.
# DDL은 Alembic migration에서 직접 발행 (SAIndex로는 partial 지원 안 함).

# ── poi_photo ─────────────────────────────────────────────────────────────────

poi_photo = sa.Table(
    "poi_photo",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column(
        "poi_mark_id",
        sa.BigInteger,
        sa.ForeignKey("poi_mark.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
    sa.Column("keyframe_seq", sa.Integer, nullable=False),
    sa.Column("captured_at", sa.BigInteger, nullable=False),
    sa.Column("bbox_x", sa.Float, nullable=True),
    sa.Column("bbox_y", sa.Float, nullable=True),
    sa.Column("bbox_w", sa.Float, nullable=True),
    sa.Column("bbox_h", sa.Float, nullable=True),
    sa.Column("class_name", sa.Text, nullable=False),  # 'manual' 리터럴 허용
    sa.Column("confidence", sa.Float, nullable=False),
    sa.Column("image_path", sa.Text, nullable=True),  # 편의 캐시
    sa.Column("image_blob", BYTEA, nullable=True),
    sa.ForeignKeyConstraint(
        ["scan_id", "keyframe_seq"],
        ["keyframe_meta.scan_id", "keyframe_meta.seq"],
        ondelete="CASCADE",
    ),
    sa.Index("ix_poi_photo_poi_mark_id", "poi_mark_id"),
    sa.Index("ix_poi_photo_scan_keyframe", "scan_id", "keyframe_seq"),
)

# ── branch_mark ───────────────────────────────────────────────────────────────

branch_mark = sa.Table(
    "branch_mark",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
    sa.Column("keyframe_seq", sa.Integer, nullable=False),
    sa.Column("created_at", sa.BigInteger, nullable=False),
    sa.Column("pose_matrix", BYTEA, nullable=False),
    sa.Column("tx", sa.Float, nullable=False),
    sa.Column("ty", sa.Float, nullable=False),
    sa.Column("tz", sa.Float, nullable=False),
    # v8 추가
    sa.Column("node_type", sa.Text, nullable=False, server_default="corridor"),
    sa.Column("width_m", sa.Float, nullable=True),
    sa.Column("connect_hint", sa.Text, nullable=True),
    sa.Column("connect_node_id", sa.Text, nullable=True),
    sa.Column("mark_session_id", sa.Text, nullable=True),
    sa.Column("dx_local", sa.Float, nullable=True),
    sa.Column("dy_local", sa.Float, nullable=True),
    sa.Column("dz_local", sa.Float, nullable=True),
    # 0011: sidecar sqlite 원본 id (branch_edge.from_node_id/to_node_id 와 매칭용).
    sa.Column("local_id", sa.Integer, nullable=True),
    sa.ForeignKeyConstraint(
        ["scan_id", "keyframe_seq"],
        ["keyframe_meta.scan_id", "keyframe_meta.seq"],
        ondelete="CASCADE",
    ),
)

# ── interfloor_mark (Sprint 65) ───────────────────────────────────────────────

interfloor_mark = sa.Table(
    "interfloor_mark",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
    sa.Column("keyframe_seq", sa.Integer, nullable=False),
    sa.Column("created_at", sa.BigInteger, nullable=False),
    sa.Column("connector_type", sa.Text, nullable=False),
    sa.Column("prefix", sa.Text, nullable=False),
    sa.Column("pose_matrix", BYTEA, nullable=False),
    sa.Column("tx", sa.Float, nullable=False),
    sa.Column("ty", sa.Float, nullable=False),
    sa.Column("tz", sa.Float, nullable=False),
    # v8 추가
    sa.Column("dx_local", sa.Float, nullable=True),
    sa.Column("dy_local", sa.Float, nullable=True),
    sa.Column("dz_local", sa.Float, nullable=True),
    sa.CheckConstraint(
        "connector_type IN ('elevator','escalator','stairs')",
        name="ck_interfloor_mark_connector_type",
    ),
    sa.ForeignKeyConstraint(
        ["scan_id", "keyframe_seq"],
        ["keyframe_meta.scan_id", "keyframe_meta.seq"],
        ondelete="CASCADE",
    ),
    sa.Index("ix_interfloor_mark_scan", "scan_id"),
)


# ── branch_edge (v9: 사용자 명시 branch_mark 끼리의 edge) ─────────────────────

branch_edge = sa.Table(
    "branch_edge",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
    sa.Column("from_node_id", sa.Text, nullable=False),  # branch_mark.id 의 string
    sa.Column("to_node_id", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),  # 'sequential' | 'cornerPolygon'
    sa.Column("length_m", sa.Float, nullable=False),
    sa.Column("mark_session_id", sa.Text, nullable=True),
    sa.Column("polygon_closed", sa.Integer, nullable=True),
    sa.Column("created_at", sa.BigInteger, nullable=False),
    sa.CheckConstraint(
        "kind IN ('sequential','cornerPolygon')",
        name="ck_branch_edge_kind",
    ),
    sa.ForeignKeyConstraint(
        ["scan_id"], ["scan_session.scan_id"], ondelete="CASCADE",
    ),
    sa.Index("ix_branch_edge_scan", "scan_id"),
)

# ── yolo_detection ────────────────────────────────────────────────────────────

yolo_detection = sa.Table(
    "yolo_detection",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
    sa.Column("keyframe_seq", sa.Integer, nullable=False),
    sa.Column("class_name", sa.Text, nullable=False),
    sa.Column("confidence", sa.Float, nullable=False),
    sa.Column("bbox_x", sa.Float, nullable=False),
    sa.Column("bbox_y", sa.Float, nullable=False),
    sa.Column("bbox_w", sa.Float, nullable=False),
    sa.Column("bbox_h", sa.Float, nullable=False),
    sa.Column("mask_rle", BYTEA, nullable=True),
    sa.Column(
        "source",
        detection_source_enum,
        nullable=False,
        server_default="on_device",
    ),
    sa.Column("track_id", sa.Integer, nullable=True),
    sa.ForeignKeyConstraint(
        ["scan_id", "keyframe_seq"],
        ["keyframe_meta.scan_id", "keyframe_meta.seq"],
        ondelete="CASCADE",
    ),
    sa.Index("ix_yolo_detection_scan_keyframe", "scan_id", "keyframe_seq"),
)

# ── build_job ─────────────────────────────────────────────────────────────────

build_job = sa.Table(
    "build_job",
    metadata,
    sa.Column("build_job_id", UUID(as_uuid=False), primary_key=True),
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column(
        "state",
        build_state_enum,
        nullable=False,
        server_default="pending",
    ),
    sa.Column("current_step", build_step_enum, nullable=True),
    sa.Column("progress", sa.Float, nullable=True),
    sa.Column(
        "enqueued_at",
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    ),
    sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column("locked_at", sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column("worker_id", sa.Text, nullable=True),
    sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
    sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
    sa.Column("failure_reason", build_failure_reason_enum, nullable=True),
    sa.Column("failure_detail", sa.Text, nullable=True),
    sa.Column("counts", JSONB, nullable=True),
    sa.UniqueConstraint("scan_id", "build_job_id"),
)

# ── map_node ──────────────────────────────────────────────────────────────────

map_node = sa.Table(
    "map_node",
    metadata,
    sa.Column("node_id", UUID(as_uuid=False), primary_key=True),
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column(
        "build_job_id",
        UUID(as_uuid=False),
        sa.ForeignKey("build_job.build_job_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("node_type", node_type_enum, nullable=False),
    sa.Column("geom", Geometry("POINTZ", srid=0), nullable=False),
    sa.Column("label", sa.Text, nullable=True),
    sa.Column(
        "poi_mark_id",
        sa.BigInteger,
        sa.ForeignKey("poi_mark.id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.Column("source_ref", JSONB, nullable=True),
    sa.Column("is_stale", sa.Boolean, nullable=False, server_default="false"),
)

# ── map_edge ──────────────────────────────────────────────────────────────────

map_edge = sa.Table(
    "map_edge",
    metadata,
    sa.Column("edge_id", UUID(as_uuid=False), primary_key=True),
    sa.Column(
        "scan_id",
        UUID(as_uuid=False),
        sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column(
        "build_job_id",
        UUID(as_uuid=False),
        sa.ForeignKey("build_job.build_job_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column(
        "from_node_id",
        UUID(as_uuid=False),
        sa.ForeignKey("map_node.node_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column(
        "to_node_id",
        UUID(as_uuid=False),
        sa.ForeignKey("map_node.node_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("edge_type", edge_type_enum, nullable=False),
    sa.Column("geom", Geometry("LINESTRINGZ", srid=0), nullable=False),
    sa.Column("length_m", sa.Float, nullable=False),
    sa.Column("is_stale", sa.Boolean, nullable=False, server_default="false"),
)
