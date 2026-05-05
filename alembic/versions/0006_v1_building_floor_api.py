"""V1 building/floor compatibility API storage.

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-05
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import BYTEA, UUID

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "building",
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

    op.create_table(
        "building_floor",
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

    op.create_table(
        "floor_scan",
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
    )
    op.create_index("ix_floor_scan_floor", "floor_scan", ["floor_id"])
    op.create_index("ix_floor_scan_scan", "floor_scan", ["scan_id"])
    op.execute(
        "CREATE UNIQUE INDEX uq_floor_scan_one_active "
        "ON floor_scan(floor_id) WHERE active = true"
    )

    op.add_column("poi_photo", sa.Column("image_blob", BYTEA(), nullable=True))
    op.add_column(
        "poi_canonical",
        sa.Column(
            "building_id",
            UUID(as_uuid=False),
            sa.ForeignKey("building.building_id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "poi_canonical",
        sa.Column(
            "floor_id",
            UUID(as_uuid=False),
            sa.ForeignKey("building_floor.floor_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_poi_canonical_building", "poi_canonical", ["building_id"])
    op.create_index("ix_poi_canonical_floor", "poi_canonical", ["floor_id"])

    op.create_foreign_key(
        "fk_vertical_connector_building",
        "vertical_connector",
        "building",
        ["building_id"],
        ["building_id"],
        ondelete="CASCADE",
    )
    op.alter_column("vertical_connector_stop", "poi_canonical_id", nullable=True)


def downgrade() -> None:
    op.alter_column("vertical_connector_stop", "poi_canonical_id", nullable=False)
    op.drop_constraint(
        "fk_vertical_connector_building",
        "vertical_connector",
        type_="foreignkey",
    )
    op.drop_index("ix_poi_canonical_floor", table_name="poi_canonical")
    op.drop_index("ix_poi_canonical_building", table_name="poi_canonical")
    op.drop_column("poi_canonical", "floor_id")
    op.drop_column("poi_canonical", "building_id")
    op.drop_column("poi_photo", "image_blob")
    op.execute("DROP INDEX IF EXISTS uq_floor_scan_one_active")
    op.drop_index("ix_floor_scan_scan", table_name="floor_scan")
    op.drop_index("ix_floor_scan_floor", table_name="floor_scan")
    op.drop_table("floor_scan")
    op.drop_table("building_floor")
    op.drop_table("building")
