"""Sidecar v8 schema 반영: branch_mark.node_type/mark_session_id/width_m,
poi/interfloor_mark.dx_local 등 + 명시적 노드↔엣지 테이블 (mark_edge).

클라가 보내는 데이터:
  - branch_mark.node_type: 'corridor' | 'corner'
  - branch_mark.width_m: corridor 폭 (NULL = polygon 제외, route 전용 노드)
  - branch_mark.mark_session_id: 같은 polygon 의 corner 그룹 키
  - branch_mark.connect_hint, connect_node_id: 명시적 연결 정보
  - poi_mark/interfloor_mark/branch_mark.dx_local/dy_local/dz_local: 카메라 local offset

새 테이블 mark_edge: 사용자가 직접 찍은 노드↔엣지.
  - corridor edge (corridor↔corridor)
  - corner edge (corner↔corner, polygon outline)
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0008"
down_revision: str | None = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── branch_mark 새 컬럼 ────────────────────────────────────────────────
    op.add_column(
        "branch_mark",
        sa.Column(
            "node_type", sa.Text(), nullable=False, server_default="corridor"
        ),
    )
    op.create_check_constraint(
        "ck_branch_mark_node_type",
        "branch_mark",
        "node_type IN ('corridor', 'corner')",
    )
    op.add_column("branch_mark", sa.Column("width_m", sa.Float(), nullable=True))
    op.add_column("branch_mark", sa.Column("connect_hint", sa.Text(), nullable=True))
    op.add_column("branch_mark", sa.Column("connect_node_id", sa.Text(), nullable=True))
    op.add_column("branch_mark", sa.Column("mark_session_id", sa.Text(), nullable=True))
    op.add_column("branch_mark", sa.Column("dx_local", sa.Float(), nullable=True))
    op.add_column("branch_mark", sa.Column("dy_local", sa.Float(), nullable=True))
    op.add_column("branch_mark", sa.Column("dz_local", sa.Float(), nullable=True))
    op.create_index(
        "ix_branch_mark_session", "branch_mark", ["mark_session_id"],
        postgresql_where=sa.text("mark_session_id IS NOT NULL"),
    )

    # ── poi_mark / interfloor_mark dx/dy/dz_local ─────────────────────────
    for tbl in ("poi_mark", "interfloor_mark"):
        op.add_column(tbl, sa.Column("dx_local", sa.Float(), nullable=True))
        op.add_column(tbl, sa.Column("dy_local", sa.Float(), nullable=True))
        op.add_column(tbl, sa.Column("dz_local", sa.Float(), nullable=True))

    # ── mark_edge 신설 ─────────────────────────────────────────────────────
    # 사용자가 찍은 명시적 edge.
    # from/to 가 어떤 mark 테이블의 row 인지 from_node_kind 로 구분.
    op.create_table(
        "mark_edge",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("scan_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("edge_kind", sa.Text(), nullable=False),  # 'corridor' | 'corner'
        sa.Column("from_node_kind", sa.Text(), nullable=False),
        # 'branch_mark' (corridor or corner) | 'poi_mark' | 'interfloor_mark'
        sa.Column("from_node_local_id", sa.BigInteger(), nullable=False),
        sa.Column("to_node_kind", sa.Text(), nullable=False),
        sa.Column("to_node_local_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["scan_id"], ["scan_session.scan_id"], ondelete="CASCADE",
        ),
    )
    op.create_check_constraint(
        "ck_mark_edge_edge_kind",
        "mark_edge",
        "edge_kind IN ('corridor', 'corner')",
    )
    op.create_check_constraint(
        "ck_mark_edge_from_kind",
        "mark_edge",
        "from_node_kind IN ('branch_mark', 'poi_mark', 'interfloor_mark')",
    )
    op.create_check_constraint(
        "ck_mark_edge_to_kind",
        "mark_edge",
        "to_node_kind IN ('branch_mark', 'poi_mark', 'interfloor_mark')",
    )
    op.create_index("ix_mark_edge_scan", "mark_edge", ["scan_id"])
    op.create_unique_constraint(
        "uq_mark_edge_pair",
        "mark_edge",
        ["scan_id", "from_node_kind", "from_node_local_id",
         "to_node_kind", "to_node_local_id", "edge_kind"],
    )


def downgrade() -> None:
    op.drop_table("mark_edge")
    for tbl in ("poi_mark", "interfloor_mark"):
        op.drop_column(tbl, "dz_local")
        op.drop_column(tbl, "dy_local")
        op.drop_column(tbl, "dx_local")
    op.drop_index("ix_branch_mark_session", table_name="branch_mark")
    for col in ("dz_local", "dy_local", "dx_local",
                "mark_session_id", "connect_node_id", "connect_hint", "width_m"):
        op.drop_column("branch_mark", col)
    op.drop_constraint("ck_branch_mark_node_type", "branch_mark", type_="check")
    op.drop_column("branch_mark", "node_type")
