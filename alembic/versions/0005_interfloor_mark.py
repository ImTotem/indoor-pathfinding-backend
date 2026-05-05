"""interfloor_mark 도메인 테이블 (Sprint 65)

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-29

변경 사항:
- interfloor_mark 테이블 신설 (계단/엘리베이터/에스컬레이터 층간 연결 노드).
- iOS sidecar v6 의 interfloor_mark 를 그대로 받아 영속화한다.
- prefix (예 "EV-A", "ST-B") 는 Sprint 62 VerticalConnectorResolver 의 connector_key 매칭에 사용.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "interfloor_mark",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("scan_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("keyframe_seq", sa.Integer, nullable=False),
        sa.Column("created_at", sa.BigInteger, nullable=False),
        sa.Column(
            "connector_type",
            sa.Text,
            nullable=False,
        ),
        sa.Column("prefix", sa.Text, nullable=False),
        sa.Column("pose_matrix", sa.LargeBinary, nullable=False),
        sa.Column("tx", sa.Float, nullable=False),
        sa.Column("ty", sa.Float, nullable=False),
        sa.Column("tz", sa.Float, nullable=False),
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


def downgrade() -> None:
    op.drop_table("interfloor_mark")
