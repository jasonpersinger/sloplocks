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
            "slimegrinder": [
                {
                    "home_team": "Knicks",
                    "away_team": "Heat",
                    "date": "2026-03-28",
                    "start_time": "2026-03-28T21:00:00Z",
                    "pick": "away",
                    "model_prob": 0.58,
                    "edge": 0.031,
                    "confidence_score": 66,
                    "american_odds": 105,
                }
            ],
            "matches": [nba_match],
            "diagnostics": {
                "summary": "modeled=1 | odds=1/1 | +ev=1 | eligible=1 | locks=1",
                "coverage_gap_examples": [],
            },
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
            "diagnostics": {
                "summary": "modeled=1 | odds=1/1 | +ev=1 | eligible=1 | locks=0",
                "coverage_gap_examples": ["Giants @ Dodgers"],
            },
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
    assert "official picks" in payload["content"]
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
    assert "missing: Giants @ Dodgers" in full_text
    assert "totals=1" in full_text
    assert "SLOP LOCK" in full_text
    assert "SLIMEGRINDER" in full_text
    assert "secondary qualified picks; stronger than radar, below official locks" in full_text
    assert "official pick" in payload["content"]


def test_build_payload_falls_back_to_radar_matches(monkeypatch, tmp_path):
    _write_predictions(
        tmp_path,
        "nhl",
        {
            "slop_locks": [],
            "longslop": None,
            "matches": [
                _base_match(
                    home_team="Bruins",
                    away_team="Rangers",
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

    assert "radar lean" in payload["content"]
    assert "MODEL RADAR" in full_text
    assert "not official picks or qualified slimegrinders" in full_text
    assert "Not an official pick" in full_text
    assert "Bruins" in full_text
    assert "Mar 28 11:00 PM ET" in full_text
    assert "SLATE DIAGNOSTICS" in full_text


def test_build_payload_includes_wnba_from_sport_order(monkeypatch, tmp_path):
    _write_predictions(
        tmp_path,
        "wnba",
        {
            "slop_locks": [],
            "longslop": None,
            "matches": [
                _base_match(
                    home_team="Liberty",
                    away_team="Fever",
                    date="2026-05-08",
                    start_time="2026-05-08T23:30:00Z",
                    confidence_score=67,
                    model_prob=0.59,
                    american_odds=-120,
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

    assert "WNBA" in full_text
    assert "Liberty" in full_text
    assert "May 08 07:30 PM ET" in full_text


def test_build_payload_suppresses_radar_for_curated_matchup(monkeypatch, tmp_path):
    """An official side should block the opposite side from appearing as radar."""
    _write_predictions(
        tmp_path,
        "nba",
        {
            "slop_locks": [
                {
                    "home_team": "Pistons",
                    "away_team": "Magic",
                    "date": "2026-05-01",
                    "start_time": "2026-05-01T23:00:00Z",
                    "pick": "home",
                    "model_prob": 0.56,
                    "edge": 0.042,
                    "expected_value": 0.08,
                    "confidence_score": 68,
                    "american_odds": 120,
                }
            ],
            "longslop": None,
            "matches": [
                _base_match(
                    home_team="Pistons",
                    away_team="Magic",
                    date="2026-05-01",
                    start_time="2026-05-01T23:00:00Z",
                    pick="away",
                    model_prob=0.57,
                    confidence_score=77,
                    american_odds=-130,
                    model_probs={"home": 0.43, "away": 0.57},
                    best_odds={"home": 120, "away": -130},
                    edges={
                        "away": {
                            "model_prob": 0.57,
                            "implied_prob": 0.565,
                            "edge": 0.005,
                            "expected_value": 0.01,
                            "american_odds": -130,
                            "fractional_kelly": 0.002,
                        }
                    },
                ),
                _base_match(
                    home_team="Knicks",
                    away_team="Heat",
                    date="2026-05-01",
                    start_time="2026-05-02T00:30:00Z",
                    pick="home",
                    model_prob=0.64,
                    confidence_score=72,
                    american_odds=-110,
                ),
            ],
            "diagnostics": {"summary": "modeled=2 | odds=2/2 | +ev=1 | eligible=1 | locks=1"},
        },
    )

    monkeypatch.setattr(notify_discord, "DATA_DIR", tmp_path)

    payload = notify_discord.build_payload()
    radar_embed = next(embed for embed in payload["embeds"] if "MODEL RADAR" in embed["title"])
    radar_text = json.dumps(radar_embed)

    assert "Pistons" not in radar_text
    assert "Magic" not in radar_text
    assert "Knicks" in radar_text
    assert "Heat" in radar_text


def test_build_payload_suppresses_slimegrinder_for_curated_matchup(monkeypatch, tmp_path):
    _write_predictions(
        tmp_path,
        "nba",
        {
            "slop_locks": [
                {
                    "home_team": "Pistons",
                    "away_team": "Magic",
                    "date": "2026-05-01",
                    "start_time": "2026-05-01T23:00:00Z",
                    "pick": "home",
                    "model_prob": 0.56,
                    "edge": 0.042,
                    "expected_value": 0.08,
                    "confidence_score": 68,
                    "american_odds": 120,
                }
            ],
            "longslop": None,
            "slimegrinder": [
                {
                    "home_team": "Pistons",
                    "away_team": "Magic",
                    "date": "2026-05-01",
                    "start_time": "2026-05-01T23:00:00Z",
                    "pick": "away",
                    "model_prob": 0.57,
                    "edge": 0.02,
                    "expected_value": 0.03,
                    "confidence_score": 66,
                    "american_odds": -130,
                },
                {
                    "home_team": "Knicks",
                    "away_team": "Heat",
                    "date": "2026-05-01",
                    "start_time": "2026-05-02T00:30:00Z",
                    "pick": "home",
                    "model_prob": 0.58,
                    "edge": 0.02,
                    "expected_value": 0.03,
                    "confidence_score": 66,
                    "american_odds": -110,
                },
            ],
            "matches": [],
            "diagnostics": {"summary": "modeled=2 | odds=2/2 | +ev=2 | eligible=1 | locks=1"},
        },
    )

    monkeypatch.setattr(notify_discord, "DATA_DIR", tmp_path)

    payload = notify_discord.build_payload()
    full_text = json.dumps(payload)

    assert "SLIMEGRINDER" in full_text
    assert "Pistons vs Magic" in full_text
    assert "Knicks vs Heat" in full_text
    slime_embed = next(embed for embed in payload["embeds"] if "SLIMEGRINDER" in embed["title"])
    slime_text = json.dumps(slime_embed)
    assert "Pistons" not in slime_text
    assert "Magic" not in slime_text
    assert "Knicks" in slime_text
    assert "Heat" in slime_text


def test_build_payload_respects_publication_guard(monkeypatch, tmp_path):
    _write_predictions(
        tmp_path,
        "nba",
        {
            "publication_guard": {
                "enforced": True,
                "allow_moneyline": False,
                "allow_totals": False,
                "status": "suppressed",
                "reason": "recent moneylines CLV is below threshold",
            },
            "slop_locks": [
                {
                    "home_team": "Pistons",
                    "away_team": "Magic",
                    "date": "2026-05-01",
                    "start_time": "2026-05-01T23:00:00Z",
                    "pick": "home",
                    "model_prob": 0.56,
                    "edge": 0.042,
                    "expected_value": 0.08,
                    "confidence_score": 68,
                    "american_odds": 120,
                }
            ],
            "totals_locks": [
                {
                    "market_type": "total",
                    "home_team": "Pistons",
                    "away_team": "Magic",
                    "date": "2026-05-01",
                    "start_time": "2026-05-01T23:00:00Z",
                    "pick": "over",
                    "total_line": 209.5,
                    "model_prob": 0.57,
                    "confidence_score": 70,
                    "american_odds": -110,
                }
            ],
            "longslop": None,
            "slimegrinder": [],
            "matches": [
                _base_match(
                    home_team="Pistons",
                    away_team="Magic",
                    date="2026-05-01",
                    start_time="2026-05-01T23:00:00Z",
                    pick="home",
                    model_prob=0.56,
                    confidence_score=68,
                    american_odds=120,
                )
            ],
            "diagnostics": {"summary": "modeled=1 | odds=1/1 | +ev=1 | eligible=1 | locks=1"},
        },
    )

    monkeypatch.setattr(notify_discord, "DATA_DIR", tmp_path)

    payload = notify_discord.build_payload()
    full_text = json.dumps(payload)

    assert "official pick" not in payload["content"]
    assert "**SLOP LOCK**" not in full_text
    assert "TOTAL LOCK" not in full_text
    assert "MODEL RADAR" in full_text
    assert "Pistons" in full_text
    assert "Magic" in full_text


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
