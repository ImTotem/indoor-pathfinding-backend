"""branch_mark.local_id 추가 — sidecar sqlite 원본 id 보존.

이전: ingest 시 sidecar 의 sqlite id (sparse, 예 1,4,5) 를 버리고 PG sequence 로
      새 id 부여. branch_edge.from_node_id/to_node_id 는 원본 sqlite id (text) 그대로 저장 →
      v2_corridor backbone 매핑 시 PG id ASC 순으로 1..N 가정해서 매핑 깨짐.
이후: PG 에 sidecar sqlite id 를 local_id 로 별도 보존 → branch_edge 매칭 정확.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0011"
down_revision: str | None = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "branch_mark",
        sa.Column("local_id", sa.Integer, nullable=True),
    )
    op.create_index(
        "ix_branch_mark_scan_local",
        "branch_mark",
        ["scan_id", "local_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_branch_mark_scan_local", table_name="branch_mark")
    op.drop_column("branch_mark", "local_id")
