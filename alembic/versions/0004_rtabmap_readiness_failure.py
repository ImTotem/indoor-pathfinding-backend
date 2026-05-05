"""rtabmap readiness failure enum

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-26

변경 사항:
- build_failure_reason enum에 rtabmap_data_not_ready 추가
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE build_failure_reason "
        "ADD VALUE IF NOT EXISTS 'rtabmap_data_not_ready'"
    )


def downgrade() -> None:
    # PostgreSQL enum value removal requires type recreation. Keep the value on
    # downgrade; old application versions simply will not emit it.
    pass
