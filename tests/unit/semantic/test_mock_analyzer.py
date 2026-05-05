from __future__ import annotations

from indoor_server.application.semantic.mock_analyzer import MockSemanticAnalyzer


def test_mock_analyzer_extracts_room_number() -> None:
    result = MockSemanticAnalyzer().analyze(label="301호 컴퓨터비전 연구실")

    assert result.category == "lab"
    assert result.name == "301호 컴퓨터비전 연구실"
    assert result.confidence >= 0.7
    assert result.connector_key is None


def test_mock_analyzer_extracts_stair_connector_key() -> None:
    result = MockSemanticAnalyzer().analyze(label="STAIR_A 동쪽 계단")

    assert result.category == "stairs"
    assert result.connector_type == "stairs"
    assert result.connector_key == "STAIR_A"


def test_mock_analyzer_extracts_elevator_connector_key() -> None:
    result = MockSemanticAnalyzer().analyze(label="ELEV_CENTER 중앙 엘리베이터")

    assert result.category == "elevator"
    assert result.connector_type == "elevator"
    assert result.connector_key == "ELEV_CENTER"
