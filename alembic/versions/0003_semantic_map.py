"""semantic map schema

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-25

변경 사항:
- poi_canonical 실제 semantic POI 필드 추가
- place_area / poi_analysis_job / place_label_candidate 추가
- vertical_connector / vertical_connector_stop 추가
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "poi_canonical",
        sa.Column(
            "scan_id",
            UUID(as_uuid=False),
            sa.ForeignKey("scan_ingest.scan_id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "poi_canonical",
        sa.Column("level_id", sa.Text, nullable=False, server_default="level-0"),
    )
    op.add_column(
        "poi_canonical",
        sa.Column("category", sa.Text, nullable=False, server_default="unknown"),
    )
    op.add_column("poi_canonical", sa.Column("name", sa.Text, nullable=True))
    op.add_column(
        "poi_canonical",
        sa.Column(
            "route_node_id",
            UUID(as_uuid=False),
            sa.ForeignKey("map_node.node_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "poi_canonical",
        sa.Column("display_point", sa.Text, nullable=True),
    )
    op.execute(
        "ALTER TABLE poi_canonical ALTER COLUMN display_point "
        "TYPE geometry(PointZ, 0) USING NULL"
    )
    op.add_column(
        "poi_canonical",
        sa.Column("display_area_id", UUID(as_uuid=False), nullable=True),
    )
    op.add_column(
        "poi_canonical",
        sa.Column("source_mark_ids", JSONB, nullable=True),
    )
    op.add_column(
        "poi_canonical",
        sa.Column("needs_review", sa.Boolean, nullable=False, server_default="false"),
    )
    op.create_index("ix_poi_canonical_scan", "poi_canonical", ["scan_id"])
    op.create_index("ix_poi_canonical_route_node", "poi_canonical", ["route_node_id"])

    op.create_table(
        "place_area",
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
        sa.Column("geom", sa.Text, nullable=False),
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
    op.execute(
        "ALTER TABLE place_area ALTER COLUMN geom "
        "TYPE geometry(MultiPolygon, 0) USING ST_GeomFromText(geom, 0)"
    )
    op.create_index("ix_place_area_scan", "place_area", ["scan_id"])
    op.execute("CREATE INDEX ix_place_area_geom ON place_area USING GIST (geom)")

    op.create_table(
        "poi_analysis_job",
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
    op.create_index("ix_poi_analysis_job_poi_mark", "poi_analysis_job", ["poi_mark_id"])

    op.create_table(
        "place_label_candidate",
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
    op.create_index(
        "ix_place_label_candidate_canonical",
        "place_label_candidate",
        ["canonical_id"],
    )

    op.create_table(
        "vertical_connector",
        sa.Column(
            "connector_id",
            UUID(as_uuid=False),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("building_id", UUID(as_uuid=False), nullable=True),
        sa.Column("connector_type", sa.Text, nullable=False),
        sa.Column("connector_key", sa.Text, nullable=False),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("building_id", "connector_type", "connector_key"),
    )

    op.create_table(
        "vertical_connector_stop",
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
            nullable=False,
        ),
        sa.Column(
            "route_node_id",
            UUID(as_uuid=False),
            sa.ForeignKey("map_node.node_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.UniqueConstraint("connector_id", "level_id"),
    )


def downgrade() -> None:
    op.drop_table("vertical_connector_stop")
    op.drop_table("vertical_connector")
    op.drop_table("place_label_candidate")
    op.drop_table("poi_analysis_job")
    op.drop_table("place_area")

    op.drop_index("ix_poi_canonical_route_node", table_name="poi_canonical")
    op.drop_index("ix_poi_canonical_scan", table_name="poi_canonical")
    op.drop_column("poi_canonical", "needs_review")
    op.drop_column("poi_canonical", "source_mark_ids")
    op.drop_column("poi_canonical", "display_area_id")
    op.drop_column("poi_canonical", "display_point")
    op.drop_column("poi_canonical", "route_node_id")
    op.drop_column("poi_canonical", "name")
    op.drop_column("poi_canonical", "category")
    op.drop_column("poi_canonical", "level_id")
    op.drop_column("poi_canonical", "scan_id")
