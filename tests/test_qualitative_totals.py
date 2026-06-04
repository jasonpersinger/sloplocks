from unittest.mock import MagicMock
import pipeline.qualitative_analysis as qa


def test_analyze_total_qualitative_returns_default_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs", "date": "2026-06-03"}
    result = qa.analyze_total_qualitative(game, "wind blowing out 15mph")
    assert result["total_impact"] == 0.0
    assert result["net_total_edge"] == "none"


def test_analyze_total_qualitative_returns_default_without_context(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs", "date": "2026-06-03"}
    result = qa.analyze_total_qualitative(game, "")
    assert result["net_total_edge"] == "none"


def test_analyze_total_qualitative_parses_model_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    parsed = qa.TotalsQualitativeAnalysis(
        sport="mlb", home_team="Reds", away_team="Cubs", total_line=9.5,
        total_impact=2.0, individual_factors=[],
        net_total_edge="over", summary="Wind out to RF favors the over.",
    )
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(parsed=parsed))]
    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse.return_value = fake_completion
    monkeypatch.setattr(qa, "OpenAI", lambda api_key: fake_client)

    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs",
            "date": "2026-06-03", "total_line": 9.5}
    result = qa.analyze_total_qualitative(game, "wind blowing out 15mph")
    assert result["net_total_edge"] == "over"
    assert result["total_impact"] == 2.0
