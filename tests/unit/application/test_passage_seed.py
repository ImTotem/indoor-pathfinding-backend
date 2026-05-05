"""Sprint 78 A-9: mock passage seed 관련 단위 테스트.

config.enable_mock_passage 설정과 is_mock 플래그 관련 동작을 검증한다.
DB 의존 없음 — pure unit.
"""
from __future__ import annotations

import pytest

from indoor_server.interfaces.api.v1_schemas import VerticalPassageResponse, PassageSegment


class TestVerticalPassageResponseMockField:
    """VerticalPassageResponse에 mock 필드가 올바르게 포함되는지 확인."""

    def test_mock_false_by_default(self) -> None:
        resp = VerticalPassageResponse(
            passageId="00000000-0000-0000-0000-000000000001",
            buildingId=None,
            connectorType="ELEVATOR",
            connectorKey="ev-1",
            name="ELEVATOR EV-1",
            segments=[],
        )
        assert resp.mock is False

    def test_mock_true_explicit(self) -> None:
        resp = VerticalPassageResponse(
            passageId="00000000-0000-0000-0000-000000000001",
            buildingId=None,
            connectorType="ELEVATOR",
            connectorKey="ev-1",
            name="ELEVATOR EV-1 (mock)",
            mock=True,
            segments=[],
        )
        assert resp.mock is True

    def test_serialized_json_includes_mock(self) -> None:
        resp = VerticalPassageResponse(
            passageId="00000000-0000-0000-0000-000000000001",
            buildingId=None,
            connectorType="ELEVATOR",
            connectorKey="ev-1",
            name=None,
            mock=True,
            segments=[],
        )
        data = resp.model_dump(by_alias=True)
        assert "mock" in data
        assert data["mock"] is True


class TestConfigMockPassageEnvGate:
    """INDOOR_SERVER_ENABLE_MOCK_PASSAGE env gate 검증."""

    def test_default_true(self) -> None:
        from indoor_server.config import Settings

        s = Settings()
        assert s.enable_mock_passage is True

    def test_env_override_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pydantic-settings는 env prefix 없이 필드명 대문자로 읽음.
        enable_mock_passage → 환경변수 ENABLE_MOCK_PASSAGE.
        production 배포 가이드에서 이 이름 사용 필요.
        """
        monkeypatch.setenv("ENABLE_MOCK_PASSAGE", "false")
        from indoor_server.config import Settings

        s = Settings()
        assert s.enable_mock_passage is False
