"""vertical_connector.is_mock 컬럼 추가 (Sprint 78 passage mock seed).

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-05
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "vertical_connector",
        sa.Column(
            "is_mock",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("vertical_connector", "is_mock")
