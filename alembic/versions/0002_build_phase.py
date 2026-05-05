"""build phase schema v2

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-21

변경 사항:
- build_state enum: queued→pending, built→succeeded, cancelled 추가
- build_job 테이블 신규 (SKIP LOCKED worker queue)
- map_node, map_edge 테이블 신규 (skeleton graph)
- 신규 enum: node_type, edge_type, build_step, build_failure_reason
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── build_state enum 갱신 ─────────────────────────────────────────────────
    # 1. 기존 값 rename (queued → pending, built → succeeded)
    op.execute("ALTER TYPE build_state RENAME VALUE 'queued' TO 'pending'")
    op.execute("ALTER TYPE build_state RENAME VALUE 'built' TO 'succeeded'")
    # 2. cancelled 추가
    op.execute("ALTER TYPE build_state ADD VALUE IF NOT EXISTS 'cancelled'")

    # ── 신규 enum 타입 ────────────────────────────────────────────────────────
    op.execute(
        "CREATE TYPE node_type AS ENUM "
        "('junction','endpoint','corridor','poi','poi_attach')"
    )
    op.execute("CREATE TYPE edge_type AS ENUM ('skeleton','poi_spur')")
    op.execute(
        "CREATE TYPE build_step AS ENUM "
        "('init','floor_seg','back_project','walkable_grid','skeleton',"
        "'node_placement','poi_projection','quality_gate','persist','done')"
    )
    op.execute(
        "CREATE TYPE build_failure_reason AS ENUM "
        "('walkable_coverage_low','graph_disconnected','model_load_failed',"
        "'intrinsics_missing','internal')"
    )

    # ── build_job ─────────────────────────────────────────────────────────────
    op.create_table(
        "build_job",
        sa.Column("build_job_id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "scan_id",
            UUID(as_uuid=False),
            sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("state", sa.Text, nullable=False, server_default="pending"),
        sa.Column("current_step", sa.Text, nullable=True),
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
        sa.Column("failure_reason", sa.Text, nullable=True),
        sa.Column("failure_detail", sa.Text, nullable=True),
        sa.Column("counts", JSONB, nullable=True),
        sa.UniqueConstraint("scan_id", "build_job_id"),
    )
    # enum 타입으로 교체 (PG 15+: TEXT default → enum 캐스팅 불가, default 먼저 drop 후 재설정)
    op.execute("ALTER TABLE build_job ALTER COLUMN state DROP DEFAULT")
    op.execute(
        "ALTER TABLE build_job ALTER COLUMN state TYPE build_state "
        "USING state::build_state"
    )
    op.execute(
        "ALTER TABLE build_job ALTER COLUMN state SET DEFAULT 'pending'::build_state"
    )
    op.execute(
        "ALTER TABLE build_job ALTER COLUMN current_step TYPE build_step "
        "USING current_step::build_step"
    )
    op.execute(
        "ALTER TABLE build_job ALTER COLUMN failure_reason TYPE build_failure_reason "
        "USING failure_reason::build_failure_reason"
    )
    # 부분 인덱스
    op.execute(
        "CREATE INDEX idx_build_job_dispatch ON build_job (enqueued_at) "
        "WHERE state = 'pending'"
    )
    op.execute(
        "CREATE INDEX idx_build_job_running_locked ON build_job (locked_at) "
        "WHERE state = 'running'"
    )
    op.execute(
        "CREATE INDEX idx_build_job_scan_recent ON build_job (scan_id, enqueued_at DESC)"
    )

    # ── map_node ──────────────────────────────────────────────────────────────
    op.create_table(
        "map_node",
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
        sa.Column("node_type", sa.Text, nullable=False),
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
    op.execute(
        "ALTER TABLE map_node ALTER COLUMN node_type TYPE node_type "
        "USING node_type::node_type"
    )
    op.execute("ALTER TABLE map_node ADD COLUMN geom geometry(PointZ, 0) NOT NULL "
               "DEFAULT ST_MakePoint(0,0,0)")
    op.execute("ALTER TABLE map_node ALTER COLUMN geom DROP DEFAULT")
    op.execute("CREATE INDEX ON map_node USING GIST (geom)")
    op.execute("CREATE INDEX ON map_node (scan_id) WHERE is_stale = false")

    # ── map_edge ──────────────────────────────────────────────────────────────
    op.create_table(
        "map_edge",
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
        sa.Column("edge_type", sa.Text, nullable=False),
        sa.Column("length_m", sa.Float, nullable=False),
        sa.Column("is_stale", sa.Boolean, nullable=False, server_default="false"),
    )
    op.execute(
        "ALTER TABLE map_edge ALTER COLUMN edge_type TYPE edge_type "
        "USING edge_type::edge_type"
    )
    op.execute(
        "ALTER TABLE map_edge ADD COLUMN geom geometry(LineStringZ, 0) NOT NULL "
        "DEFAULT ST_GeomFromText('LINESTRINGZ(0 0 0, 1 1 0)', 0)"
    )
    op.execute("ALTER TABLE map_edge ALTER COLUMN geom DROP DEFAULT")
    op.execute("CREATE INDEX ON map_edge USING GIST (geom)")
    op.execute("CREATE INDEX ON map_edge (scan_id) WHERE is_stale = false")
    op.execute("CREATE INDEX ON map_edge (from_node_id)")
    op.execute("CREATE INDEX ON map_edge (to_node_id)")


def downgrade() -> None:
    op.drop_table("map_edge")
    op.drop_table("map_node")
    op.drop_table("build_job")

    op.execute("DROP TYPE IF EXISTS build_failure_reason")
    op.execute("DROP TYPE IF EXISTS build_step")
    op.execute("DROP TYPE IF EXISTS edge_type")
    op.execute("DROP TYPE IF EXISTS node_type")

    # build_state enum을 Sprint 15 원래 값으로 되돌리기
    # PostgreSQL은 enum value rename/remove를 직접 지원하지 않으므로
    # 타입 재생성 방식 사용
    op.execute(
        "ALTER TYPE build_state RENAME TO build_state_old"
    )
    op.execute(
        "CREATE TYPE build_state AS ENUM ('not_started','queued','running','built','failed')"
    )
    # scan_ingest 컬럼 타입 교체
    op.execute(
        "ALTER TABLE scan_ingest ALTER COLUMN build_state TYPE build_state "
        "USING CASE build_state::text "
        "  WHEN 'pending' THEN 'queued'::build_state "
        "  WHEN 'succeeded' THEN 'built'::build_state "
        "  WHEN 'cancelled' THEN 'failed'::build_state "
        "  ELSE build_state::text::build_state "
        "END"
    )
    op.execute("DROP TYPE build_state_old")
