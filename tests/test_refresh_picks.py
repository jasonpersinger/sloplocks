"""Tests for the late-day refresh path."""

import json

from pipeline import refresh_picks


def _write_predictions(base_dir, sport, payload):
    sport_dir = base_dir / "data" / sport
    sport_dir.mkdir(parents=True, exist_ok=True)
    with open(sport_dir / "predictions.json", "w") as f:
        json.dump(payload, f)


def test_merge_live_fixture_moves_pitcher_provenance_as_a_bundle():
    record = {
        "home_team": "Aces",
        "away_team": "Bruins",
        "date": "2026-07-04",
        "home_pitcher": "Old Home",
        "home_pitcher_hand": "R",
        "home_pitcher_source": "espn",
        "home_pitcher_last_checked": "2026-07-04T10:00:00+00:00",
        "home_pitcher_cache_stale": False,
        "away_pitcher": "Old Away",
        "away_pitcher_hand": "L",
        "away_pitcher_source": "espn",
        "away_pitcher_last_checked": "2026-07-04T10:00:00+00:00",
        "away_pitcher_cache_stale": False,
        "pitcher_warnings": [],
    }
    live_fixture = {
        "home_pitcher": "New Home",
        "home_pitcher_hand": "L",
        "home_pitcher_source": "mlb_stats_api",
        "home_pitcher_last_checked": "2026-07-04T14:00:00+00:00",
        "home_pitcher_cache_stale": True,
        "away_pitcher": "New Away",
        "away_pitcher_hand": "R",
        "away_pitcher_source": "mlb_stats_api",
        "away_pitcher_last_checked": "2026-07-04T14:00:00+00:00",
        "away_pitcher_cache_stale": True,
        "pitcher_warnings": ["home_pitcher_from_stale_cache", "away_pitcher_from_stale_cache"],
    }

    merged = refresh_picks._merge_live_fixture(record, live_fixture)

    assert merged["home_pitcher"] == "New Home"
    assert merged["home_pitcher_hand"] == "L"
    assert merged["home_pitcher_source"] == "mlb_stats_api"
    assert merged["home_pitcher_last_checked"] == "2026-07-04T14:00:00+00:00"
    assert merged["home_pitcher_cache_stale"] is True
    assert merged["away_pitcher"] == "New Away"
    assert merged["away_pitcher_hand"] == "R"
    assert merged["away_pitcher_source"] == "mlb_stats_api"
    assert merged["away_pitcher_last_checked"] == "2026-07-04T14:00:00+00:00"
    assert merged["away_pitcher_cache_stale"] is True
    assert merged["pitcher_warnings"] == ["home_pitcher_from_stale_cache", "away_pitcher_from_stale_cache"]


def test_merge_live_fixture_does_not_update_pitcher_name_without_provenance():
    record = {
        "home_pitcher": "Old Home",
        "home_pitcher_hand": "R",
        "home_pitcher_source": "espn",
        "home_pitcher_last_checked": "2026-07-04T10:00:00+00:00",
        "away_pitcher": "Old Away",
        "away_pitcher_hand": "L",
        "away_pitcher_source": "espn",
        "away_pitcher_last_checked": "2026-07-04T10:00:00+00:00",
    }

    merged = refresh_picks._merge_live_fixture(
        record,
        {"home_pitcher": "Unproven Home", "away_pitcher": "Unproven Away"},
    )

    assert merged["home_pitcher"] == "Old Home"
    assert merged["away_pitcher"] == "Old Away"


def test_refresh_nba_updates_live_metadata_and_preserves_baseline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payload = {
        "outcomes": ["home", "away"],
        "model_weights": {"elo": 1.0},
        "matches": [
            {
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": "2026-03-29",
                "start_time": "2026-03-29T00:30:00Z",
                "model_probs": {"home": 0.5, "away": 0.5},
                "individual_models": {"elo": {"home": 0.5, "away": 0.5}},
                "pick": "home",
                "edges": {},
            }
        ],
        "totals_matches": [
            {
                "market_type": "total",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "date": "2026-03-29",
                "start_time": "2026-03-29T00:30:00Z",
                "expected_total": 224.0,
                "total_stddev": 13.5,
                "individual_models": {"totals_model": {"over": 0.55, "under": 0.45}},
            }
        ],
        "slop_locks": [],
        "totals_locks": [],
        "slimegrinder": [],
        "pick_stats": {"all": {"total": 0}},
        "diagnostics": {"historical_matches": 100},
    }
    _write_predictions(tmp_path, "nba", payload)

    monkeypatch.setattr(
        refresh_picks,
        "fetch_nba_espn_schedule",
        lambda cache_path=None: [{
            "home_team": "Lakers",
            "away_team": "Warriors",
            "date": "2026-03-29",
            "start_time": "2026-03-29T03:00:00Z",
            "home_availability_profile": {
                "active_players": 15,
                "available_core_players": 12,
                "injury_burden": 0.0,
                "key_absence_score": 0.0,
                "leader_absence_burden": 0.0,
            },
            "away_availability_profile": {
                "active_players": 13,
                "available_core_players": 8,
                "injury_burden": 1.0,
                "key_absence_score": 1.0,
                "leader_absence_burden": 1.2,
            },
        }],
    )
    monkeypatch.setattr(
        refresh_picks,
        "fetch_odds",
        lambda sport_key, include_totals=False: [{
            "home_team": "Lakers",
            "away_team": "Warriors",
            "commence_time": "2026-03-29T03:00:00Z",
            "home_odds": 2.05,
            "away_odds": 1.82,
            "total_line": 223.5,
            "over_odds": 1.91,
            "under_odds": 1.91,
        }],
    )
    monkeypatch.setattr(refresh_picks, "_append_odds_snapshot_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_load_latest_odds_snapshots", lambda *args, **kwargs: {})
    monkeypatch.setattr(refresh_picks, "_apply_latest_market_snapshots", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_save_json", lambda path, data: None)

    refresh_picks.refresh_sport("nba")

    with open(tmp_path / "data" / "nba" / "predictions.json") as f:
        data = json.load(f)

    match = data["matches"][0]
    assert match["start_time"] == "2026-03-29T03:00:00Z"
    assert match["base_model_probs"] == {"home": 0.5, "away": 0.5}
    assert match["model_probs"]["home"] > 0.5
    assert match["home_availability_profile"]["available_core_players"] == 12
    assert data["totals_matches"][0]["base_expected_total"] == 224.0
    assert data["diagnostics"]["historical_matches"] == 100
    assert data["run_type"] == "refresh"
    assert data["snapshot_path"].endswith(".json")
    assert (tmp_path / "data" / data["snapshot_path"]).exists()


def test_refresh_wnba_normalizes_odds_and_updates_availability(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payload = {
        "outcomes": ["home", "away"],
        "model_weights": {"elo": 1.0},
        "matches": [
            {
                "home_team": "Liberty",
                "away_team": "Fever",
                "date": "2026-05-08",
                "start_time": "2026-05-08T23:30:00Z",
                "model_probs": {"home": 0.5, "away": 0.5},
                "individual_models": {"elo": {"home": 0.5, "away": 0.5}},
                "pick": "home",
                "edges": {},
            }
        ],
        "totals_matches": [],
        "slop_locks": [],
        "totals_locks": [],
        "slimegrinder": [],
        "pick_stats": {"all": {"total": 0}},
        "diagnostics": {"historical_matches": 40},
    }
    _write_predictions(tmp_path, "wnba", payload)

    monkeypatch.setattr(
        refresh_picks,
        "fetch_wnba_espn_schedule",
        lambda cache_path=None: [{
            "home_team": "Liberty",
            "away_team": "Fever",
            "date": "2026-05-08",
            "start_time": "2026-05-08T23:30:00Z",
            "home_availability_profile": {
                "active_players": 12,
                "available_core_players": 10,
                "injury_burden": 0.0,
                "key_absence_score": 0.0,
                "leader_absence_burden": 0.0,
            },
            "away_availability_profile": {
                "active_players": 11,
                "available_core_players": 7,
                "injury_burden": 1.0,
                "key_absence_score": 1.0,
                "leader_absence_burden": 1.0,
            },
        }],
    )
    monkeypatch.setattr(
        refresh_picks,
        "fetch_odds",
        lambda sport_key, include_totals=False: [{
            "home_team": "New York Liberty",
            "away_team": "Indiana Fever",
            "commence_time": "2026-05-08T23:30:00Z",
            "home_odds": 1.9,
            "away_odds": 2.0,
            "draw_odds": 0.0,
        }],
    )
    monkeypatch.setattr(refresh_picks, "_append_odds_snapshot_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_load_latest_odds_snapshots", lambda *args, **kwargs: {})
    monkeypatch.setattr(refresh_picks, "_apply_latest_market_snapshots", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_save_json", lambda path, data: None)

    refresh_picks.refresh_sport("wnba")

    with open(tmp_path / "data" / "wnba" / "predictions.json") as f:
        data = json.load(f)

    match = data["matches"][0]
    assert match["base_model_probs"] == {"home": 0.5, "away": 0.5}
    assert match["model_probs"]["home"] > 0.5
    assert match["edges"]["home"]["american_odds"] < 0
    assert data["diagnostics"]["historical_matches"] == 40


def test_refresh_nhl_updates_goalie_metadata(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payload = {
        "outcomes": ["home", "away"],
        "model_weights": {"elo": 1.0},
        "matches": [
            {
                "home_team": "Bruins",
                "away_team": "Canadiens",
                "date": "2026-03-29",
                "model_probs": {"home": 0.52, "away": 0.48},
                "individual_models": {"elo": {"home": 0.52, "away": 0.48}},
                "pick": "home",
                "edges": {},
            }
        ],
        "totals_matches": [],
        "slop_locks": [],
        "totals_locks": [],
        "slimegrinder": [],
        "diagnostics": {"historical_matches": 50},
    }
    _write_predictions(tmp_path, "nhl", payload)

    monkeypatch.setattr(
        refresh_picks,
        "fetch_nhl_schedule",
        lambda cache_path=None: [{
            "home_team": "Bruins",
            "away_team": "Canadiens",
            "date": "2026-03-29",
            "start_time": "2026-03-29T23:00:00Z",
            "home_goalie": "Linus Ullmark",
            "away_goalie": "Sam Montembeault",
            "home_goalie_status": "confirmed",
            "away_goalie_status": "projected",
        }],
    )
    monkeypatch.setattr(
        refresh_picks,
        "fetch_odds",
        lambda sport_key, include_totals=False: [{
            "home_team": "Bruins",
            "away_team": "Canadiens",
            "commence_time": "2026-03-29T23:00:00Z",
            "home_odds": 1.9,
            "away_odds": 1.95,
            "draw_odds": 0.0,
        }],
    )
    monkeypatch.setattr(refresh_picks, "_append_odds_snapshot_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_load_latest_odds_snapshots", lambda *args, **kwargs: {})
    monkeypatch.setattr(refresh_picks, "_apply_latest_market_snapshots", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_save_json", lambda path, data: None)

    refresh_picks.refresh_sport("nhl")

    with open(tmp_path / "data" / "nhl" / "predictions.json") as f:
        data = json.load(f)

    match = data["matches"][0]
    assert match["home_goalie"] == "Linus Ullmark"
    assert match["away_goalie_status"] == "projected"
    assert match["base_model_probs"] == {"home": 0.52, "away": 0.48}


def test_refresh_skips_invalid_live_odds_without_crashing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payload = {
        "outcomes": ["home", "away"],
        "model_weights": {"elo": 1.0},
        "matches": [
            {
                "home_team": "Bruins",
                "away_team": "Rangers",
                "date": "2026-03-29",
                "model_probs": {"home": 0.52, "away": 0.48},
                "individual_models": {"elo": {"home": 0.52, "away": 0.48}},
                "pick": "home",
                "edges": {},
            }
        ],
        "totals_matches": [],
        "slop_locks": [],
        "totals_locks": [],
        "slimegrinder": [],
        "diagnostics": {"historical_matches": 20},
    }
    _write_predictions(tmp_path, "nhl", payload)
    monkeypatch.setattr(
        refresh_picks,
        "fetch_nhl_schedule",
        lambda cache_path=None: [{
            "home_team": "Bruins",
            "away_team": "Rangers",
            "date": "2026-03-29",
            "start_time": "2026-03-29T23:00:00Z",
        }],
    )
    monkeypatch.setattr(refresh_picks, "normalize_nhl_team_name", lambda name: name)

    monkeypatch.setattr(
        refresh_picks,
        "fetch_odds",
        lambda sport_key, include_totals=False: [{
            "home_team": "Bruins",
            "away_team": "Rangers",
            "commence_time": "2026-03-29T23:00:00Z",
            "home_odds": 1.0,
            "away_odds": 1.92,
        }],
    )
    monkeypatch.setattr(refresh_picks, "_append_odds_snapshot_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_load_latest_odds_snapshots", lambda *args, **kwargs: {})
    monkeypatch.setattr(refresh_picks, "_apply_latest_market_snapshots", lambda *args, **kwargs: None)
    monkeypatch.setattr(refresh_picks, "_save_json", lambda path, data: None)

    refresh_picks.refresh_sport("nhl")

    with open(tmp_path / "data" / "nhl" / "predictions.json") as f:
        data = json.load(f)

    assert data["matches"][0]["edges"]["away"]["american_odds"] < 0


def test_refresh_skips_season_disabled_sport(capsys):
    result = refresh_picks.refresh_sport("ncaam")

    assert result is None
    assert "season-disabled" in capsys.readouterr().out


def test_main_continues_when_one_sport_refresh_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    calls = []

    def fake_refresh(sport_key, run_context=None):
        calls.append(sport_key)
        if sport_key == "nhl":
            raise RuntimeError("bad live payload")

    monkeypatch.setattr(refresh_picks, "refresh_sport", fake_refresh)
    monkeypatch.setattr(refresh_picks, "build_dashboard_data", lambda base_dir: {"ok": True})
    monkeypatch.setattr(refresh_picks, "_save_json", lambda path, data: None)
    monkeypatch.setattr(refresh_picks, "SPORTS", {"nba": {}, "nhl": {}})
    monkeypatch.setattr(refresh_picks, "sys", type("SysProxy", (), {"argv": ["refresh_picks.py"]}))

    refresh_picks.main()

    out = capsys.readouterr().out
    assert calls == ["nba", "nhl"]
    assert "Refresh failures:" in out
    assert "nhl: bad live payload" in out
