"""node_type enum 에 passage_stairs / passage_elevator / passage_escalator 추가.

이전: 층간연결도 NodeType.POI 로 저장 + label/source_ref.connector_type 으로만 구분.
이후: enum 자체로 sub-type 분리 → 관리자 viewer / 클라가 type 만 보고 분기 가능.

기존 데이터: source_ref.role == 'vertical_connector_stop' 인 POI 노드를
            source_ref.connector_type 에 따라 PASSAGE_* 로 마이그레이션.
"""
from __future__ import annotations

from alembic import op


revision: str = "0010"
down_revision: str | None = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE node_type ADD VALUE IF NOT EXISTS 'passage_stairs'")
    op.execute("ALTER TYPE node_type ADD VALUE IF NOT EXISTS 'passage_elevator'")
    op.execute("ALTER TYPE node_type ADD VALUE IF NOT EXISTS 'passage_escalator'")
    op.execute("COMMIT")  # ENUM ADD 는 트랜잭션 밖이어야 후속 UPDATE 에서 사용 가능
    op.execute(
        """
        UPDATE map_node
        SET node_type = CASE source_ref->>'connector_type'
            WHEN 'stairs' THEN 'passage_stairs'::node_type
            WHEN 'elevator' THEN 'passage_elevator'::node_type
            WHEN 'escalator' THEN 'passage_escalator'::node_type
            ELSE node_type
        END
        WHERE node_type = 'poi'
          AND source_ref->>'role' = 'vertical_connector_stop'
          AND source_ref->>'connector_type' IN ('stairs', 'elevator', 'escalator')
        """
    )


def downgrade() -> None:
    # PASSAGE_* → POI 로 되돌림 (enum 값 자체는 PG 가 drop 못 함, 데이터만 정리).
    op.execute(
        """
        UPDATE map_node SET node_type = 'poi'
        WHERE node_type IN ('passage_stairs', 'passage_elevator', 'passage_escalator')
        """
    )
