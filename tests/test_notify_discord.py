"""Tests for pipeline.notify_discord."""

import json

import pipeline.notify_discord as notify_discord


def _write_predictions(tmp_path, sport, payload):
    sport_dir = tmp_path / sport
    sport_dir.mkdir(parents=True, exist_ok=True)
    with open(sport_dir / "predictions.json", "w") as f:
        json.dump(payload, f)


def _write_dashboard(tmp_path, payload):
    with open(tmp_path / "dashboard.json", "w") as f:
        json.dump(payload, f)


def _base_match(**overrides):
    match = {
        "home_team": "Lakers",
        "away_team": "Celtics",
        "date": "2026-03-28",
        "start_time": "2026-03-28T23:00:00Z",
        "pick": "home",
        "model_prob": 0.62,
        "confidence_score": 74,
        "edge": 0.061,
        "american_odds": 115,
        "model_probs": {"home": 0.62, "away": 0.38},
        "best_odds": {"home": 115, "away": -125},
        "edges": {
            "home": {
                "model_prob": 0.62,
                "implied_prob": 0.559,
                "edge": 0.061,
                "expected_value": 0.173,
                "american_odds": 115,
                "fractional_kelly": 0.041,
            }
        },
        "completed": False,
    }
    match.update(overrides)
    return match


def test_build_payload_includes_all_sports_and_richer_fields(monkeypatch, tmp_path):
    nba_match = _base_match()
    nba_lock = {
        "home_team": "Lakers",
        "away_team": "Celtics",
        "date": "2026-03-28",
        "start_time": "2026-03-28T23:00:00Z",
        "pick": "home",
        "model_prob": 0.62,
        "edge": 0.061,
        "confidence_score": 74,
        "american_odds": 115,
        "blurb": "The numbers like the Lakers here.",
    }
    _write_predictions(
        tmp_path,
        "nba",
        {
            "slop_locks": [nba_lock],
            "longslop": None,
            "matches": [nba_match],
            "diagnostics": {"summary": "modeled=1 | odds=1/1 | +ev=1 | eligible=1 | locks=1"},
        },
    )

    mlb_match = _base_match(
        home_team="Dodgers",
        away_team="Padres",
        pick="away",
        model_prob=0.29,
        confidence_score=68,
        edge=0.034,
        american_odds=525,
        model_probs={"home": 0.71, "away": 0.29},
        best_odds={"home": -650, "away": 525},
        edges={
            "away": {
                "model_prob": 0.29,
                "implied_prob": 0.256,
                "edge": 0.034,
                "expected_value": 0.812,
                "american_odds": 525,
                "fractional_kelly": 0.019,
            }
        },
        home_pitcher="Yamamoto",
        away_pitcher="Cease",
    )
    mlb_longslop = {
        "home_team": "Dodgers",
        "away_team": "Padres",
        "date": "2026-03-28",
        "start_time": "2026-03-29T02:00:00Z",
        "pick": "away",
    }
    mlb_total_lock = {
        "market_type": "total",
        "home_team": "Dodgers",
        "away_team": "Padres",
        "date": "2026-03-28",
        "start_time": "2026-03-29T02:00:00Z",
        "pick": "over",
        "total_line": 8.5,
        "expected_total": 9.4,
        "model_prob": 0.59,
        "implied_prob": 0.512,
        "edge": 0.048,
        "expected_value": 0.14,
        "american_odds": -105,
        "fractional_kelly": 0.018,
        "confidence_score": 61,
    }
    _write_predictions(
        tmp_path,
        "mlb",
        {
            "slop_locks": [],
            "totals_locks": [mlb_total_lock],
            "longslop": mlb_longslop,
            "matches": [mlb_match],
            "diagnostics": {"summary": "modeled=1 | odds=1/1 | +ev=1 | eligible=1 | locks=0"},
        },
    )
    _write_dashboard(
        tmp_path,
        {
            "recommended_actions": [
                {"priority": "high", "title": "Fix Live Coverage Gaps", "detail": "Odds coverage is missing in NHL 5/6."}
            ]
        },
    )

    monkeypatch.setattr(notify_discord, "DATA_DIR", tmp_path)

    payload = notify_discord.build_payload()
    full_text = json.dumps(payload)

    assert payload["username"] == "BIG SLIME"
    assert "curated picks" in payload["content"]
    assert "Lakers" in full_text
    assert "Dodgers" in full_text
    assert "OVER 8.5" in full_text
    assert "Proj 9.4" in full_text
    assert "LONGSLOP" in full_text
    assert "EV 0.17u" in full_text
    assert "Kelly 4.1%" in full_text
    assert "Pitchers: Cease vs Yamamoto" in full_text
    assert "Mar 28 07:00 PM ET" in full_text
    assert "SLATE DIAGNOSTICS" in full_text
    assert "CONTROL PANEL" in full_text
    assert "Fix Live Coverage Gaps" in full_text
    assert "modeled=1 | odds=1/1 | +ev=1 | eligible=1 | locks=1" in full_text
    assert "totals=1" in full_text
    assert "SLOP LOCK" in full_text


def test_build_payload_falls_back_to_radar_matches(monkeypatch, tmp_path):
    _write_predictions(
        tmp_path,
        "mma",
        {
            "slop_locks": [],
            "longslop": None,
            "matches": [
                _base_match(
                    home_team="Jon Jones",
                    away_team="Tom Aspinall",
                    start_time="2026-03-29T03:00:00Z",
                    confidence_score=71,
                    model_prob=0.64,
                    american_odds=-135,
                    edges={},
                    best_odds={},
                )
            ],
            "diagnostics": {"summary": "modeled=1 | odds=1/1 | +ev=1 | eligible=0 | locks=0"},
        },
    )

    monkeypatch.setattr(notify_discord, "DATA_DIR", tmp_path)

    payload = notify_discord.build_payload()
    full_text = json.dumps(payload)

    assert "radar spot" in payload["content"]
    assert "MODEL RADAR" in full_text
    assert "Jon Jones" in full_text
    assert "Mar 28 11:00 PM ET" in full_text
    assert "SLATE DIAGNOSTICS" in full_text


def test_build_payload_handles_empty_day(monkeypatch, tmp_path):
    _write_predictions(
        tmp_path,
        "ncaam",
        {
            "slop_locks": [],
            "longslop": None,
            "matches": [],
        },
    )

    monkeypatch.setattr(notify_discord, "DATA_DIR", tmp_path)

    payload = notify_discord.build_payload()

    assert payload["embeds"] == []
    assert "No picks qualified today" in payload["content"]
