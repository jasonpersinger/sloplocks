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


from pipeline.run import (
    _apply_total_qualitative_adjustment,
    _format_total_qualitative_summary,
)


def test_total_nudge_positive_raises_total():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 3.0}, weight=0.4, max_points_delta=0.5)
    assert out > 9.0


def test_total_nudge_negative_lowers_total():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": -3.0}, weight=0.4, max_points_delta=0.5)
    assert out < 9.0


def test_total_nudge_respects_cap():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 5.0}, weight=999.0, max_points_delta=0.5)
    assert out == 9.5  # clamped to +max_points_delta


def test_total_nudge_zero_impact_is_noop():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 0.0}, weight=0.4, max_points_delta=0.5)
    assert out == 9.0


def test_total_summary_none_edge_is_neutral():
    summary = _format_total_qualitative_summary(
        {"over": 0.5, "under": 0.5},
        {"net_total_edge": "none", "total_impact": 0.0, "individual_factors": []})
    assert summary == "No qualitative impact."


def test_total_summary_over_edge_mentions_over():
    summary = _format_total_qualitative_summary(
        {"over": 0.6, "under": 0.4},
        {"net_total_edge": "over", "total_impact": 2.0,
         "individual_factors": [{"description": "wind out to RF"}]})
    assert "Over" in summary
    assert "wind out to RF" in summary
