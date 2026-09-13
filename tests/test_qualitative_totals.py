import json
from unittest.mock import MagicMock

import pipeline.qualitative_analysis as qa


def test_analyze_total_qualitative_returns_default_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs", "date": "2026-06-03"}
    result = qa.analyze_total_qualitative(game, "wind blowing out 15mph")
    assert result["total_impact"] == 0.0
    assert result["net_total_edge"] == "none"


def test_analyze_total_qualitative_returns_default_without_context(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs", "date": "2026-06-03"}
    result = qa.analyze_total_qualitative(game, "")
    assert result["net_total_edge"] == "none"


def _fake_gemini(monkeypatch, payload: dict):
    """Stub the Gemini client so no network call is made."""
    response = MagicMock()
    response.text = json.dumps(payload)
    client = MagicMock()
    client.models.generate_content.return_value = response
    monkeypatch.setattr(qa.genai, "Client", lambda **kwargs: client)
    return client


def test_analyze_total_qualitative_parses_model_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    _fake_gemini(monkeypatch, {
        "sport": "mlb", "home_team": "Reds", "away_team": "Cubs", "total_line": 9.5,
        "total_impact": 2.0, "individual_factors": [],
        "net_total_edge": "over", "summary": "Wind out to RF favors the over.",
        "pick_rationale": "The projection sits above the posted line.",
        "model_vs_market": "Model 10.2 vs a posted 9.5.",
        "key_risk": "The wind dies down at first pitch.",
        "confidence_label": "moderate",
    })

    game = {"sport": "mlb", "home_team": "Reds", "away_team": "Cubs",
            "date": "2026-06-03", "total_line": 9.5}
    result = qa.analyze_total_qualitative(game, "wind blowing out 15mph")
    assert result["net_total_edge"] == "over"
    assert result["total_impact"] == 2.0
    assert result["pick_rationale"]
    assert result["key_risk"]


def test_analyze_game_qualitative_parses_model_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    _fake_gemini(monkeypatch, {
        "sport": "nfl", "home_team": "Chiefs", "away_team": "Broncos",
        "home_impact": 0.0, "away_impact": -2.0, "individual_factors": [],
        "net_qualitative_edge": "home", "summary": "Broncos missing a starter.",
        "pick_rationale": "The model prices the Chiefs above the book.",
        "model_vs_market": "62% model vs 55% implied.",
        "key_risk": "A short week blunts the home edge.",
        "confidence_label": "strong",
    })

    game = {"sport": "nfl", "home_team": "Chiefs", "away_team": "Broncos",
            "date": "2026-09-13", "pick": "home", "model_prob": 0.62,
            "implied_prob": 0.55, "edge": 0.07}
    result = qa.analyze_game_qualitative(game, "Broncos LT ruled out")
    assert result["net_qualitative_edge"] == "home"
    assert result["pick_rationale"]
    assert result["confidence_label"] == "strong"


def test_gemini_failure_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("boom")
    monkeypatch.setattr(qa.genai, "Client", lambda **kwargs: client)

    game = {"sport": "nfl", "home_team": "Chiefs", "away_team": "Broncos",
            "date": "2026-09-13"}
    result = qa.analyze_game_qualitative(game, "some context")
    assert result["home_impact"] == 0.0
    assert result["net_qualitative_edge"] == "none"


def test_game_analysis_runs_without_situational_context(monkeypatch):
    """College football has no injury feed; the model output still deserves a why."""
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    client = _fake_gemini(monkeypatch, {
        "sport": "ncaaf", "home_team": "Georgia", "away_team": "Kentucky",
        "home_impact": 0.0, "away_impact": 0.0, "individual_factors": [],
        "net_qualitative_edge": "none", "summary": "No situational signal.",
        "pick_rationale": "Elo separates these programs by a wide margin.",
        "model_vs_market": "68% model vs 64% implied.",
        "key_risk": "A backup quarterback changes the projection.",
        "confidence_label": "moderate",
    })

    game = {"sport": "ncaaf", "home_team": "Georgia", "away_team": "Kentucky",
            "date": "2026-09-12", "pick": "home", "model_prob": 0.68}
    result = qa.analyze_game_qualitative(game, "")

    assert client.models.generate_content.called
    assert result["pick_rationale"]
    # The analyst must be told not to invent news it was never given.
    sent = client.models.generate_content.call_args.kwargs["contents"]
    assert "do not speculate" in sent


def test_game_analysis_still_bails_with_nothing_to_explain(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    client = _fake_gemini(monkeypatch, {})
    result = qa.analyze_game_qualitative(
        {"sport": "ncaaf", "home_team": "A", "away_team": "B"}, ""
    )
    assert not client.models.generate_content.called
    assert result["net_qualitative_edge"] == "none"


def test_totals_analysis_runs_without_situational_context(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    client = _fake_gemini(monkeypatch, {
        "sport": "ncaaf", "home_team": "Georgia", "away_team": "Kentucky",
        "total_line": 52.5, "total_impact": 0.0, "individual_factors": [],
        "net_total_edge": "under", "summary": "Projection below the line.",
        "pick_rationale": "Both defenses have been conceding under the line.",
        "model_vs_market": "Model 48.1 vs a posted 52.5.",
        "key_risk": "A shootout script.",
        "confidence_label": "moderate",
    })
    result = qa.analyze_total_qualitative(
        {"sport": "ncaaf", "home_team": "Georgia", "away_team": "Kentucky",
         "total_line": 52.5, "expected_total": 48.1}, ""
    )
    assert client.models.generate_content.called
    assert result["pick_rationale"]


def test_model_context_includes_edge_and_components():
    rendered = qa._format_model_context({
        "pick": "home", "home_team": "Chiefs", "away_team": "Broncos",
        "model_prob": 0.62, "implied_prob": 0.55, "edge": 0.07,
        "confidence_score": 66,
        "individual_models": {"elo": {"home": 0.64}, "results_features": {"home": 0.60}},
    })
    assert "Model pick: Chiefs" in rendered
    assert "62.0%" in rendered
    assert "elo: 64.0%" in rendered


def test_totals_model_context_reports_gap_to_line():
    rendered = qa._format_totals_model_context({
        "expected_total": 47.6, "total_line": 44.5, "over_prob": 0.61, "stddev": 10.5,
    })
    assert "47.6" in rendered
    assert "+3.1" in rendered


from pipeline.run import (
    _apply_total_qualitative_adjustment,
    _format_total_qualitative_summary,
)


def test_total_nudge_positive_raises_total():
    # impact 2.5 of 5 -> half the cap: +0.25 runs
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 2.5}, max_points_delta=0.5)
    assert out == 9.25


def test_total_nudge_negative_lowers_total():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": -2.5}, max_points_delta=0.5)
    assert out == 8.75


def test_total_nudge_full_conviction_reaches_cap():
    # impact +/-5 reaches the full configured cap
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 5.0}, max_points_delta=0.5)
    assert out == 9.5


def test_total_nudge_out_of_range_impact_is_capped():
    # an out-of-range score cannot exceed the cap
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 99.0}, max_points_delta=0.5)
    assert out == 9.5


def test_total_nudge_zero_impact_is_noop():
    out = _apply_total_qualitative_adjustment(
        9.0, {"total_impact": 0.0}, max_points_delta=0.5)
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
