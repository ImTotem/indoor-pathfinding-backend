"""sidecar v9 정식 schema 반영: mark_edge → branch_edge 로 이름 변경 + 클라 컬럼 정합.

클라 v9 의 실제 branch_edge 테이블:
  CREATE TABLE branch_edge (
    id              INTEGER PRIMARY KEY,
    scan_id         TEXT,
    from_node_id    TEXT NOT NULL,        -- branch_mark.id 의 string 표현
    to_node_id      TEXT NOT NULL,
    kind            TEXT NOT NULL,        -- 'sequential' | 'cornerPolygon'
    length_m        DOUBLE NOT NULL,
    mark_session_id TEXT,
    polygon_closed  INTEGER,              -- 1=closed cycle
    created_at      INTEGER
  )

0008 에서 만든 mark_edge 는 사용 안 됐으므로 drop + branch_edge 로 신규 생성.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0009"
down_revision: str | None = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0008 에서 만든 mark_edge 제거 (실제 클라가 안 보냄)
    op.drop_table("mark_edge")

    # 새 branch_edge 테이블 — 클라 schema 정합
    op.create_table(
        "branch_edge",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("scan_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("from_node_id", sa.Text(), nullable=False),    # branch_mark.id 의 string
        sa.Column("to_node_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("length_m", sa.Float(), nullable=False),
        sa.Column("mark_session_id", sa.Text(), nullable=True),
        sa.Column("polygon_closed", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["scan_id"], ["scan_session.scan_id"], ondelete="CASCADE",
        ),
    )
    op.create_check_constraint(
        "ck_branch_edge_kind",
        "branch_edge",
        "kind IN ('sequential','cornerPolygon')",
    )
    op.create_index("ix_branch_edge_scan", "branch_edge", ["scan_id"])
    op.create_index(
        "ix_branch_edge_session", "branch_edge", ["mark_session_id"],
        postgresql_where=sa.text("mark_session_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_branch_edge_session", table_name="branch_edge")
    op.drop_index("ix_branch_edge_scan", table_name="branch_edge")
    op.drop_constraint("ck_branch_edge_kind", "branch_edge", type_="check")
    op.drop_table("branch_edge")

    # 0008 의 mark_edge 복원 (downgrade 시)
    op.create_table(
        "mark_edge",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("scan_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("edge_kind", sa.Text(), nullable=False),
        sa.Column("from_node_kind", sa.Text(), nullable=False),
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
        "edge_kind IN ('corridor','corner')",
    )
    op.create_index("ix_mark_edge_scan", "mark_edge", ["scan_id"])
