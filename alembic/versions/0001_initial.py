"""initial schema v1

Revision ID: 0001
Revises:
Create Date: 2026-04-21

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── Extensions ───────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ── Enums ─────────────────────────────────────────────────────────────────
    op.execute("CREATE TYPE poi_source AS ENUM ('track_lock', 'manual')")
    op.execute("CREATE TYPE detection_source AS ENUM ('on_device', 'server')")
    op.execute(
        "CREATE TYPE build_state AS ENUM ('not_started','queued','running','built','failed')"
    )

    # ── scan_ingest ───────────────────────────────────────────────────────────
    op.create_table(
        "scan_ingest",
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
            sa.Text,  # 임시 Text로 생성 후 아래에서 enum 타입으로 교체
            nullable=False,
            server_default="not_started",
        ),
        sa.Column("build_job_id", UUID(as_uuid=False), nullable=True),
    )

    # W-3/W-5 수정: build_state 컬럼을 build_state enum 타입으로 교체
    # (create_table에서 enum 타입 직접 사용 시 Alembic이 type 객체를 올바르게 처리하지 못하는
    #  이슈를 우회하기 위해 Text로 생성 후 ALTER COLUMN으로 교체)
    # PG 15+: TEXT default를 enum 타입으로 자동 캐스팅하지 않으므로 default를 먼저 떨어뜨린다
    op.execute("ALTER TABLE scan_ingest ALTER COLUMN build_state DROP DEFAULT")
    op.execute(
        "ALTER TABLE scan_ingest "
        "ALTER COLUMN build_state TYPE build_state "
        "USING build_state::build_state"
    )
    # server_default도 enum 캐스팅 포함 표현으로 재설정
    op.execute(
        "ALTER TABLE scan_ingest "
        "ALTER COLUMN build_state SET DEFAULT 'not_started'::build_state"
    )

    # ── scan_session ──────────────────────────────────────────────────────────
    op.create_table(
        "scan_session",
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

    # ── keyframe_meta ─────────────────────────────────────────────────────────
    op.create_table(
        "keyframe_meta",
        sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["scan_id"], ["scan_session.scan_id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_keyframe_meta_scan_captured", "keyframe_meta", ["scan_id", "captured_at"])

    # ── poi_canonical (R-4 placeholder) ──────────────────────────────────────
    op.create_table(
        "poi_canonical",
        sa.Column(
            "canonical_id",
            UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("label", sa.Text, nullable=True),
        sa.Column("world_pose", sa.Text, nullable=True),  # GEOMETRY 타입은 raw SQL로
        sa.Column("cluster_method", sa.Text, nullable=True),
        sa.Column("llm_confidence", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # GEOMETRY 컬럼은 PostGIS DDL로 별도 추가
    op.execute(
        "ALTER TABLE poi_canonical ALTER COLUMN world_pose TYPE geometry(PointZ, 0) "
        "USING world_pose::geometry"
    )
    # NULL로 시작하므로 CAST 불필요 — 먼저 DROP하고 PostGIS 타입으로 재추가
    op.execute("ALTER TABLE poi_canonical DROP COLUMN world_pose")
    op.execute(
        "ALTER TABLE poi_canonical ADD COLUMN world_pose geometry(PointZ, 0)"
    )

    # ── poi_mark ──────────────────────────────────────────────────────────────
    op.create_table(
        "poi_mark",
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
        sa.Column(
            "source",
            sa.Text,
            nullable=False,
        ),
        sa.Column("canonical_id", UUID(as_uuid=False), nullable=True),
        sa.ForeignKeyConstraint(
            ["scan_id", "keyframe_seq"],
            ["keyframe_meta.scan_id", "keyframe_meta.seq"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["canonical_id"], ["poi_canonical.canonical_id"], ondelete="SET NULL"
        ),
    )
    # source enum 타입으로 교체 (default 없음)
    op.execute("ALTER TABLE poi_mark ALTER COLUMN source TYPE poi_source USING source::poi_source")
    # R-2 world_pose placeholder
    op.execute("ALTER TABLE poi_mark ADD COLUMN world_pose geometry(PointZ, 0)")
    # 부분 UNIQUE 인덱스 — iOS v3 정책 승계
    op.execute(
        "CREATE UNIQUE INDEX poi_mark_scan_track_unique "
        "ON poi_mark(scan_id, track_id) WHERE track_id IS NOT NULL"
    )

    # ── poi_photo ─────────────────────────────────────────────────────────────
    op.create_table(
        "poi_photo",
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
        sa.Column("class_name", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("image_path", sa.Text, nullable=True),
        sa.ForeignKeyConstraint(
            ["scan_id", "keyframe_seq"],
            ["keyframe_meta.scan_id", "keyframe_meta.seq"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_poi_photo_poi_mark_id", "poi_photo", ["poi_mark_id"])
    op.create_index("ix_poi_photo_scan_keyframe", "poi_photo", ["scan_id", "keyframe_seq"])

    # ── branch_mark ───────────────────────────────────────────────────────────
    op.create_table(
        "branch_mark",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("scan_id", UUID(as_uuid=False), nullable=False),
        sa.Column("keyframe_seq", sa.Integer, nullable=False),
        sa.Column("created_at", sa.BigInteger, nullable=False),
        sa.Column("pose_matrix", BYTEA, nullable=False),
        sa.Column("tx", sa.Float, nullable=False),
        sa.Column("ty", sa.Float, nullable=False),
        sa.Column("tz", sa.Float, nullable=False),
        sa.ForeignKeyConstraint(
            ["scan_id", "keyframe_seq"],
            ["keyframe_meta.scan_id", "keyframe_meta.seq"],
            ondelete="CASCADE",
        ),
    )

    # ── yolo_detection ────────────────────────────────────────────────────────
    op.create_table(
        "yolo_detection",
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
            sa.Text,
            nullable=False,
            server_default="on_device",
        ),
        sa.Column("track_id", sa.Integer, nullable=True),
        sa.ForeignKeyConstraint(
            ["scan_id", "keyframe_seq"],
            ["keyframe_meta.scan_id", "keyframe_meta.seq"],
            ondelete="CASCADE",
        ),
    )
    op.execute("ALTER TABLE yolo_detection ALTER COLUMN source DROP DEFAULT")
    op.execute(
        "ALTER TABLE yolo_detection ALTER COLUMN source TYPE detection_source "
        "USING source::detection_source"
    )
    op.execute(
        "ALTER TABLE yolo_detection ALTER COLUMN source "
        "SET DEFAULT 'on_device'::detection_source"
    )
    op.create_index(
        "ix_yolo_detection_scan_keyframe", "yolo_detection", ["scan_id", "keyframe_seq"]
    )


def downgrade() -> None:
    op.drop_table("yolo_detection")
    op.drop_table("branch_mark")
    op.drop_table("poi_photo")
    op.drop_table("poi_mark")
    op.drop_table("poi_canonical")
    op.drop_table("keyframe_meta")
    op.drop_table("scan_session")
    op.drop_table("scan_ingest")
    op.execute("DROP TYPE IF EXISTS poi_source")
    op.execute("DROP TYPE IF EXISTS detection_source")
    op.execute("DROP TYPE IF EXISTS build_state")
