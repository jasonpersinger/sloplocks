"""Tests for NFL and NCAAF activation and football-specific model handling."""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.config import SEASON_DISABLED_SPORTS, SPORTS
from pipeline.models import EloRatings, ResultsFeatureModel
from pipeline.run import _rest_adjustment, run_sport_pipeline

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

class TestFootballActivation:
    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_sport_is_active(self, sport_key):
        """A sport is ACTIVE precisely when it is a live key in SPORTS."""
        assert sport_key in SPORTS
        assert sport_key not in SEASON_DISABLED_SPORTS

    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_has_required_pipeline_config(self, sport_key):
        sport = SPORTS[sport_key]
        for key in (
            "name", "display_name", "odds_sport", "outcomes", "models",
            "elo_k_factor", "elo_home_advantage", "accuracy_window", "data_dir",
        ):
            assert key in sport, f"{sport_key} missing {key}"
        assert sport["outcomes"] == ["home", "away"]
        assert sport["data_dir"].endswith(sport_key)

    def test_odds_api_sport_keys(self):
        assert SPORTS["nfl"]["odds_sport"] == "americanfootball_nfl"
        assert SPORTS["ncaaf"]["odds_sport"] == "americanfootball_ncaaf"

    def test_existing_sports_remain_active(self):
        for sport_key in ("nba", "nhl", "wnba", "mlb"):
            assert sport_key in SPORTS
        assert "ncaam" in SEASON_DISABLED_SPORTS

    def test_ncaaf_is_configured_distinctly_from_nfl(self):
        """NCAAF must not be NFL with different team names."""
        nfl, ncaaf = SPORTS["nfl"], SPORTS["ncaaf"]
        # Wider talent spread and a larger home edge than the pros.
        assert ncaaf["elo_k_factor"] > nfl["elo_k_factor"]
        assert ncaaf["elo_home_advantage"] > nfl["elo_home_advantage"]
        # Blowouts are routine, so college caps the rating-relevant margin.
        assert ncaaf.get("elo_margin_cap") is not None
        assert nfl.get("elo_margin_cap") is None
        # More teams and more games means a larger fitting floor.
        assert ncaaf["results_feature_min_games"] > nfl["results_feature_min_games"]
        assert ncaaf["slop_lock_edge_threshold"] > nfl["slop_lock_edge_threshold"]

    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_totals_are_enabled(self, sport_key):
        """Football totals run off FootballTotalsModel."""
        sport = SPORTS[sport_key]
        assert sport["totals_enabled"] is True
        assert sport["totals_max_picks"] > 0
        assert sport["totals_projection_min"] < sport["totals_projection_max"]

    def test_ncaaf_totals_band_is_wider_than_nfl(self):
        """College scoring is higher and more variable."""
        assert SPORTS["ncaaf"]["totals_projection_max"] > SPORTS["nfl"]["totals_projection_max"]
        assert SPORTS["ncaaf"]["totals_default_stddev"] > SPORTS["nfl"]["totals_default_stddev"]

    def test_totals_odds_flag_preserved_for_existing_sports(self):
        """nba and mlb requested totals odds before; nhl and wnba did not."""
        assert SPORTS["nba"]["totals_enabled"] is True
        assert SPORTS["mlb"]["totals_enabled"] is True
        assert SPORTS["nhl"].get("totals_enabled", False) is False
        assert SPORTS["wnba"].get("totals_enabled", False) is False

    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_football_publishes_a_full_slate(self, sport_key):
        """Both football sports show a pick on every game, not just locks."""
        assert SPORTS[sport_key]["publish_full_slate"] is True

    def test_other_sports_do_not_publish_a_full_slate(self):
        for sport_key in ("nba", "nhl", "wnba", "mlb"):
            assert SPORTS[sport_key].get("publish_full_slate", False) is False

    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_registered_in_notifier_and_normalizers(self, sport_key):
        from pipeline.notify_discord import SPORT_EMOJIS, SPORT_ORDER
        from pipeline.refresh_picks import _NORMALIZERS

        assert sport_key in SPORT_ORDER
        assert sport_key in SPORT_EMOJIS
        assert sport_key in _NORMALIZERS

    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_frontend_exposes_sport(self, sport_key):
        with open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
        ) as f:
            html = f.read()
        assert f'data-sport="{sport_key}"' in html
        assert f"'{sport_key}'" in html


# ---------------------------------------------------------------------------
# Elo margin scaling
# ---------------------------------------------------------------------------

class TestEloMarginScaling:
    def test_default_behaviour_is_unchanged(self):
        """Goal-scale sports must be bit-for-bit identical to before."""
        elo = EloRatings(["A", "B"])
        assert elo.margin_divisor == 1.0
        assert elo._margin_multiplier(1) == 1.0
        assert elo._margin_multiplier(2) == 1.5
        assert elo._margin_multiplier(4) == pytest.approx(15.0 / 8.0)

    def test_football_margin_is_measured_in_touchdowns(self):
        elo = EloRatings(["A", "B"], margin_divisor=7.0)
        # A one-score game moves ratings like a one-goal game.
        assert elo._margin_multiplier(3) == 1.0
        assert elo._margin_multiplier(7) == 1.0
        assert elo._margin_multiplier(14) == 1.5
        assert elo._margin_multiplier(21) == pytest.approx(14.0 / 8.0)

    def test_multiplier_is_monotonic_in_margin(self):
        elo = EloRatings(["A", "B"], margin_divisor=7.0)
        values = [elo._margin_multiplier(m) for m in range(0, 60)]
        assert values == sorted(values)

    def test_margin_cap_limits_blowout_influence(self):
        elo = EloRatings(["A", "B"], margin_divisor=7.0, margin_cap=28)
        assert elo._margin_multiplier(56) == elo._margin_multiplier(28)

    def test_unscaled_blowout_would_swing_more_than_scaled(self):
        """The reason football needs scaling: a 28-point win on the raw scale."""
        raw = EloRatings(["A", "B"])
        scaled = EloRatings(["A", "B"], margin_divisor=7.0)
        assert raw._margin_multiplier(28) > 2 * scaled._margin_multiplier(28)

    def test_scaled_ratings_move_less_on_a_blowout(self):
        raw = EloRatings(["A", "B"], k_factor=20)
        scaled = EloRatings(["A", "B"], k_factor=20, margin_divisor=7.0)
        raw.update("A", "B", 35, 7)
        scaled.update("A", "B", 35, 7)
        assert scaled.get_rating("A") < raw.get_rating("A")
        assert scaled.get_rating("A") > 1500.0


# ---------------------------------------------------------------------------
# Cross-season carryover
# ---------------------------------------------------------------------------

class TestSeasonCarryover:
    @staticmethod
    def _two_seasons():
        """Two seasons where the same team wins every game."""
        rows = []
        for season, base in ((2025, "2025-09-07"), (2026, "2026-09-06")):
            start = datetime.strptime(base, "%Y-%m-%d")
            for week in range(10):
                rows.append({
                    "date": (start + timedelta(days=7 * week)).strftime("%Y-%m-%d"),
                    "home_team": "Strong" if week % 2 else "Weak",
                    "away_team": "Weak" if week % 2 else "Strong",
                    "home_goals": 28 if week % 2 else 10,
                    "away_goals": 10 if week % 2 else 28,
                    "season_year": season,
                })
        return pd.DataFrame(rows)

    def test_disabled_by_default(self):
        elo = EloRatings(["A", "B"])
        assert elo.season_carryover is None

    def test_regress_to_mean_is_a_noop_at_one(self):
        elo = EloRatings(["A"])
        elo.ratings["A"] = 1700.0
        elo.regress_to_mean(1.0)
        assert elo.ratings["A"] == 1700.0

    def test_regress_to_mean_pulls_toward_initial_rating(self):
        elo = EloRatings(["A"])
        elo.ratings["A"] = 1700.0
        elo.regress_to_mean(0.5)
        assert elo.ratings["A"] == 1600.0

    def test_regress_to_mean_ignores_none(self):
        elo = EloRatings(["A"])
        elo.ratings["A"] = 1700.0
        elo.regress_to_mean(None)
        assert elo.ratings["A"] == 1700.0

    def test_boundary_detected_from_season_year_column(self):
        df = self._two_seasons()
        elo = EloRatings(["Strong", "Weak"], season_carryover=0.5)
        dates = pd.to_datetime(df.sort_values("date")["date"])
        flags = elo._season_boundaries(df.sort_values("date").reset_index(drop=True), dates)
        assert sum(flags) == 1

    def test_boundary_detected_from_date_gap_without_season_column(self):
        df = self._two_seasons().drop(columns=["season_year"])
        elo = EloRatings(["Strong", "Weak"], season_carryover=0.5)
        sorted_df = df.sort_values("date").reset_index(drop=True)
        flags = elo._season_boundaries(sorted_df, pd.to_datetime(sorted_df["date"]))
        assert sum(flags) == 1

    def test_no_boundaries_when_carryover_disabled(self):
        df = self._two_seasons()
        elo = EloRatings(["Strong", "Weak"])
        sorted_df = df.sort_values("date").reset_index(drop=True)
        flags = elo._season_boundaries(sorted_df, pd.to_datetime(sorted_df["date"]))
        assert not any(flags)

    def test_carryover_retains_prior_season_strength(self):
        """The whole point: a team enters the new season above 1500."""
        df = self._two_seasons()
        with_carry = EloRatings(["Strong", "Weak"], k_factor=20, season_carryover=0.67)
        with_carry.process_season(df)
        assert with_carry.get_rating("Strong") > with_carry.get_rating("Weak")

    def test_more_carryover_preserves_more_separation(self):
        df = self._two_seasons()
        spreads = {}
        for carry in (0.0, 0.67, 1.0):
            elo = EloRatings(["Strong", "Weak"], k_factor=20, season_carryover=carry)
            elo.process_season(df)
            spreads[carry] = elo.get_rating("Strong") - elo.get_rating("Weak")
        assert spreads[0.0] < spreads[0.67] < spreads[1.0]

    def test_process_season_unchanged_for_single_season_sports(self):
        """Existing sports must produce byte-identical ratings."""
        df = self._two_seasons().drop(columns=["season_year"])
        baseline = EloRatings(["Strong", "Weak"], k_factor=20)
        baseline.process_season(df)
        explicit_none = EloRatings(["Strong", "Weak"], k_factor=20, season_carryover=None)
        explicit_none.process_season(df)
        assert baseline.ratings == explicit_none.ratings

    @pytest.mark.parametrize("sport_key,expected", [("nfl", 0.67), ("ncaaf", 0.72)])
    def test_football_configures_carryover(self, sport_key, expected):
        assert SPORTS[sport_key]["elo_season_carryover"] == expected

    def test_college_retains_more_than_the_nfl(self):
        """No draft or salary cap, so program strength persists harder."""
        assert SPORTS["ncaaf"]["elo_season_carryover"] > SPORTS["nfl"]["elo_season_carryover"]

    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_football_pulls_multiple_seasons_of_history(self, sport_key):
        assert SPORTS[sport_key]["history_seasons"] >= 2

    def test_other_sports_do_not_opt_in(self):
        for sport_key in ("nba", "nhl", "wnba", "mlb"):
            assert SPORTS[sport_key].get("elo_season_carryover") is None
            assert SPORTS[sport_key].get("history_seasons") is None


class TestMultiSeasonFetchRange:
    def test_single_season_is_the_default(self):
        from pipeline.fetch_nfl import _season_date_range

        dates = _season_date_range(2026, history_seasons=1)
        assert min(dates) >= "2026-08-01"

    def test_history_seasons_extends_backwards(self):
        from pipeline.fetch_nfl import _season_date_range

        dates = _season_date_range(2026, history_seasons=3)
        assert min(dates) == "2024-08-01"

    def test_ncaaf_range_starts_at_week_zero(self):
        from pipeline.fetch_ncaaf import _season_date_range

        dates = _season_date_range(2026, history_seasons=3)
        assert min(dates) == "2024-08-20"

    def test_incremental_fetch_backfills_older_seasons(self):
        """Raising history_seasons must actually refetch the earlier span."""
        from pipeline.fetch_nfl import _incremental_dates

        cache = {"games": {"1": {"date": "2026-09-05"}}}
        all_dates = ["2024-09-01", "2025-09-01", "2026-09-04", "2026-09-05", "2026-09-06"]
        result = _incremental_dates(cache, all_dates, lookback_days=3)
        assert "2024-09-01" in result
        assert "2025-09-01" in result
        assert "2026-09-06" in result

    def test_incremental_fetch_skips_the_settled_middle(self):
        from pipeline.fetch_nfl import _incremental_dates

        cache = {"games": {"1": {"date": "2026-09-01"}, "2": {"date": "2026-09-20"}}}
        all_dates = ["2026-09-01", "2026-09-10", "2026-09-20"]
        result = _incremental_dates(cache, all_dates, lookback_days=3)
        assert "2026-09-10" not in result


# ---------------------------------------------------------------------------
# Weekly-cadence rest handling
# ---------------------------------------------------------------------------

class TestResultsFeatureRestCap:
    @staticmethod
    def _logs(last_date):
        return [{"date": last_date, "result": 1.0, "margin": 3.0, "venue": "home"}]

    def test_default_cap_is_one_week(self):
        model = ResultsFeatureModel(pd.DataFrame(), feature_window=5, min_games=1)
        assert model.rest_cap_days == 7.0
        # A bye week and a normal week are indistinguishable at the default cap.
        assert model._days_since_last(self._logs("2026-09-01"), "2026-09-08") == 7.0
        assert model._days_since_last(self._logs("2026-09-01"), "2026-09-15") == 7.0

    def test_football_cap_preserves_the_bye_week_signal(self):
        model = ResultsFeatureModel(
            pd.DataFrame(), feature_window=5, min_games=1, rest_cap_days=14.0
        )
        normal = model._days_since_last(self._logs("2026-09-01"), "2026-09-08")
        bye = model._days_since_last(self._logs("2026-09-01"), "2026-09-15")
        assert normal == 7.0
        assert bye == 14.0
        assert bye > normal

    def test_football_sports_configure_the_longer_cap(self):
        for sport_key in ("nfl", "ncaaf"):
            assert SPORTS[sport_key]["results_feature_rest_cap_days"] == 14


class TestFootballRestAdjustment:
    @staticmethod
    def _matches(last_game_date):
        return pd.DataFrame([{
            "date": last_game_date,
            "home_team": "Chiefs",
            "away_team": "Broncos",
            "home_goals": 24,
            "away_goals": 17,
        }])

    def test_short_week_is_penalised(self):
        sport = SPORTS["nfl"]
        # Sunday game, then Thursday: four days of rest.
        adjustment = _rest_adjustment(
            "Chiefs", "2026-09-17", self._matches("2026-09-13"), sport
        )
        assert adjustment == -sport["short_rest_penalty"]

    def test_normal_week_is_neutral(self):
        adjustment = _rest_adjustment(
            "Chiefs", "2026-09-20", self._matches("2026-09-13"), SPORTS["nfl"]
        )
        assert adjustment == 0.0

    def test_bye_week_earns_the_rest_bonus(self):
        sport = SPORTS["nfl"]
        adjustment = _rest_adjustment(
            "Chiefs", "2026-09-27", self._matches("2026-09-13"), sport
        )
        assert adjustment == sport["rest_bonus_points"]

    def test_season_opening_gap_is_not_treated_as_a_bye(self):
        """Months since the last game must not read as extra rest."""
        adjustment = _rest_adjustment(
            "Chiefs", "2026-09-10", self._matches("2026-01-05"), SPORTS["nfl"]
        )
        assert adjustment == 0.0

    def test_daily_sports_are_unaffected_by_the_new_keys(self):
        nba = SPORTS["nba"]
        assert "short_rest_days" not in nba
        assert "rest_bonus_max_days" not in nba
        matches = pd.DataFrame([{
            "date": "2026-02-10",
            "home_team": "Lakers",
            "away_team": "Warriors",
            "home_goals": 110,
            "away_goals": 104,
        }])
        # Back-to-back still penalised exactly as before.
        assert _rest_adjustment("Lakers", "2026-02-11", matches, nba) == -nba["back_to_back_penalty"]


# ---------------------------------------------------------------------------
# ESPN request shape
# ---------------------------------------------------------------------------

class TestEspnRequestsUseSingleDates:
    """ESPN's scoreboard endpoint rejects YYYYMMDD-YYYYMMDD ranges with a 400.

    Every other fetcher in this repo requests one date at a time; the football
    fetchers briefly batched whole weeks into a range, which ESPN accepted until
    2026-09-16 and then stopped accepting. These tests pin the request shape.
    """

    @staticmethod
    def _dates_params(mock_get):
        """Return the `dates` query value of every scoreboard call made."""
        from urllib.parse import parse_qs, urlparse

        values = []
        for call in mock_get.call_args_list:
            url = call.args[0] if call.args else call.kwargs.get("url", "")
            if "scoreboard" not in url:
                continue
            values.extend(parse_qs(urlparse(url).query).get("dates", []))
        return values

    @staticmethod
    def _ok(payload):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        return resp

    @patch("pipeline.fetch_nfl._team_map", {})
    @patch("pipeline.fetch_nfl.requests.get")
    def test_nfl_schedule_requests_one_date_at_a_time(self, mock_get):
        from pipeline.fetch_nfl import fetch_nfl_schedule

        mock_get.return_value = self._ok({"events": []})
        fetch_nfl_schedule()

        dates = self._dates_params(mock_get)
        assert dates, "expected at least one scoreboard request"
        for value in dates:
            assert "-" not in value, f"range request would 400: dates={value}"
            assert len(value) == 8, f"expected YYYYMMDD, got {value}"

    @patch("pipeline.fetch_nfl._team_map", {})
    @patch("pipeline.fetch_nfl.requests.get")
    def test_nfl_games_request_one_date_at_a_time(self, mock_get):
        from pipeline.fetch_nfl import fetch_nfl_games

        mock_get.return_value = self._ok({"events": []})
        fetch_nfl_games(dates=["2026-09-17", "2026-09-18", "2026-09-19"])

        dates = self._dates_params(mock_get)
        assert dates
        for value in dates:
            assert "-" not in value, f"range request would 400: dates={value}"

    @patch("pipeline.fetch_ncaaf._team_map", {})
    @patch("pipeline.fetch_ncaaf._team_conferences", {})
    @patch("pipeline.fetch_ncaaf.requests.get")
    def test_ncaaf_schedule_requests_one_date_at_a_time(self, mock_get):
        from pipeline.fetch_ncaaf import fetch_ncaaf_schedule

        mock_get.return_value = self._ok({"events": []})
        fetch_ncaaf_schedule()

        dates = self._dates_params(mock_get)
        assert dates
        for value in dates:
            assert "-" not in value, f"range request would 400: dates={value}"

    @patch("pipeline.fetch_ncaaf._team_map", {})
    @patch("pipeline.fetch_ncaaf._team_conferences", {})
    @patch("pipeline.fetch_ncaaf.requests.get")
    def test_ncaaf_games_request_one_date_at_a_time(self, mock_get):
        from pipeline.fetch_ncaaf import fetch_ncaaf_games

        mock_get.return_value = self._ok({"events": []})
        fetch_ncaaf_games(dates=["2026-09-17", "2026-09-18", "2026-09-19"])

        dates = self._dates_params(mock_get)
        assert dates
        for value in dates:
            assert "-" not in value, f"range request would 400: dates={value}"

    @patch("pipeline.fetch_nfl._team_map", {})
    @patch("pipeline.fetch_nfl.requests.get")
    def test_schedule_covers_the_whole_pipeline_window(self, mock_get):
        """run_sport_pipeline keeps yesterday through two days out."""
        from pipeline.fetch_nfl import fetch_nfl_schedule

        mock_get.return_value = self._ok({"events": []})
        fetch_nfl_schedule()

        assert len(set(self._dates_params(mock_get))) == 4


# ---------------------------------------------------------------------------
# Football totals model
# ---------------------------------------------------------------------------

class TestFootballTotalsModel:
    @staticmethod
    def _scoring_games(high_scoring, low_scoring, weeks=12):
        """Two clusters: one that plays shootouts, one that plays rock fights."""
        from itertools import product

        base = datetime.strptime("2026-09-06", "%Y-%m-%d")
        rows = []
        for week in range(weeks):
            date_str = (base + timedelta(days=7 * week)).strftime("%Y-%m-%d")
            for home, away in product(high_scoring, low_scoring):
                rows.append({
                    "date": date_str,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": 35,
                    "away_goals": 10,
                })
            for i in range(len(high_scoring) - 1):
                rows.append({
                    "date": date_str,
                    "home_team": high_scoring[i],
                    "away_team": high_scoring[i + 1],
                    "home_goals": 38,
                    "away_goals": 34,
                })
            for i in range(len(low_scoring) - 1):
                rows.append({
                    "date": date_str,
                    "home_team": low_scoring[i],
                    "away_team": low_scoring[i + 1],
                    "home_goals": 13,
                    "away_goals": 9,
                })
        return pd.DataFrame(rows)

    def _model(self, min_games=40):
        from pipeline.models import FootballTotalsModel

        games = self._scoring_games(["Air", "Sky", "Bomb"], ["Mud", "Grind", "Rock"])
        return FootballTotalsModel(games, feature_window=5, min_games=min_games), games

    def test_baseline_is_learned_from_the_data(self):
        model, games = self._model()
        expected = float((games["home_goals"] + games["away_goals"]).mean())
        assert model.baseline_total == pytest.approx(expected)

    def test_unfitted_model_returns_the_league_baseline(self):
        from pipeline.models import FootballTotalsModel

        model = FootballTotalsModel(pd.DataFrame(), min_games=10)
        assert model.predict_total({"home_team": "A", "away_team": "B"}) == model.baseline_total

    def test_min_games_gate_prevents_fitting(self):
        model, _ = self._model(min_games=10_000)
        assert model.model is None

    def test_shootout_matchup_projects_higher_than_rock_fight(self):
        model, _ = self._model()
        assert model.model is not None
        high = model.predict_total({"home_team": "Air", "away_team": "Sky"})
        low = model.predict_total({"home_team": "Mud", "away_team": "Grind"})
        assert high > low

    def test_projection_is_clamped_to_the_configured_band(self):
        from pipeline.models import FootballTotalsModel

        games = self._scoring_games(["Air", "Sky", "Bomb"], ["Mud", "Grind", "Rock"])
        model = FootballTotalsModel(
            games, feature_window=5, min_games=40,
            projection_min=40.0, projection_max=41.0,
        )
        value = model.predict_total({"home_team": "Air", "away_team": "Sky"})
        assert 40.0 <= value <= 41.0

    def test_predict_market_returns_complementary_probabilities(self):
        from pipeline.models import football_totals_predict

        model, _ = self._model()
        out = football_totals_predict(model, {"home_team": "Air", "away_team": "Sky"}, 48.5)
        assert out["over"] + out["under"] == pytest.approx(1.0)
        assert 0.0 < out["over"] < 1.0
        assert out["stddev"] >= 6.0

    def test_a_line_above_the_projection_favours_the_under(self):
        from pipeline.models import football_totals_predict

        model, _ = self._model()
        projection = model.predict_total({"home_team": "Mud", "away_team": "Grind"})
        out = football_totals_predict(
            model, {"home_team": "Mud", "away_team": "Grind"}, projection + 12.0
        )
        assert out["under"] > out["over"]

    def test_missing_score_columns_leaves_model_unfitted(self):
        from pipeline.models import FootballTotalsModel

        df = pd.DataFrame([{"date": "2026-09-06", "home_team": "A", "away_team": "B"}])
        model = FootballTotalsModel(df, min_games=1)
        assert model.model is None


# ---------------------------------------------------------------------------
# NCAAF-specific parsing
# ---------------------------------------------------------------------------

def _ncaaf_event(home, away, home_score, away_score, neutral=False):
    return {
        "id": "401",
        "date": "2026-09-05T23:00Z",
        "season": {"year": 2026, "type": 2},
        "week": {"number": 2},
        "competitions": [{
            "neutralSite": neutral,
            "status": {"type": {"completed": True}, "period": 4},
            "competitors": [
                {"homeAway": "home", "score": str(home_score), "team": {"displayName": home}},
                {"homeAway": "away", "score": str(away_score), "team": {"displayName": away}},
            ],
        }],
    }


class TestNcaafParsing:
    @pytest.fixture(autouse=True)
    def _team_tables(self, monkeypatch):
        from pipeline import fetch_ncaaf

        monkeypatch.setattr(fetch_ncaaf, "_team_map", {
            "Alabama Crimson Tide": "Alabama",
            "Georgia Bulldogs": "Georgia",
            "Ohio State Buckeyes": "Ohio State",
        })
        monkeypatch.setattr(fetch_ncaaf, "_team_conferences", {
            "Alabama": "8",
            "Georgia": "8",
            "Ohio State": "5",
        })

    def test_non_fbs_opponent_collapses_to_one_entity(self):
        from pipeline.fetch_ncaaf import FCS_OPPONENT, _parse_final_event

        parsed = _parse_final_event(
            _ncaaf_event("Alabama Crimson Tide", "Mercer Bears", 55, 3)
        )
        assert parsed["home_team"] == "Alabama"
        assert parsed["away_team"] == FCS_OPPONENT
        assert parsed["fcs_matchup"] is True

    def test_two_non_fbs_teams_are_dropped(self):
        from pipeline.fetch_ncaaf import _parse_final_event

        assert _parse_final_event(_ncaaf_event("Mercer Bears", "Samford Bulldogs", 21, 20)) is None

    def test_conference_game_is_detected(self):
        from pipeline.fetch_ncaaf import _parse_final_event

        parsed = _parse_final_event(
            _ncaaf_event("Alabama Crimson Tide", "Georgia Bulldogs", 27, 24)
        )
        assert parsed["conference_game"] is True
        assert parsed["home_conference"] == parsed["away_conference"] == "8"

    def test_non_conference_game_is_detected(self):
        from pipeline.fetch_ncaaf import _parse_final_event

        parsed = _parse_final_event(
            _ncaaf_event("Alabama Crimson Tide", "Ohio State Buckeyes", 27, 24)
        )
        assert parsed["conference_game"] is False

    def test_neutral_site_is_carried_through(self):
        from pipeline.fetch_ncaaf import _parse_final_event

        parsed = _parse_final_event(
            _ncaaf_event("Alabama Crimson Tide", "Georgia Bulldogs", 27, 24, neutral=True)
        )
        assert parsed["neutral"] is True

    def test_incomplete_game_is_skipped(self):
        from pipeline.fetch_ncaaf import _parse_final_event

        event = _ncaaf_event("Alabama Crimson Tide", "Georgia Bulldogs", 0, 0)
        event["competitions"][0]["status"]["type"]["completed"] = False
        assert _parse_final_event(event) is None


class TestNflParsing:
    @pytest.fixture(autouse=True)
    def _team_table(self, monkeypatch):
        from pipeline import fetch_nfl

        monkeypatch.setattr(fetch_nfl, "_team_map", {
            "Kansas City Chiefs": "Chiefs",
            "Denver Broncos": "Broncos",
        })

    @staticmethod
    def _event(season_type=2, period=4, home_score=24, away_score=17):
        return {
            "id": "1",
            "date": "2026-09-13T17:00Z",
            "season": {"year": 2026, "type": season_type},
            "week": {"number": 1},
            "competitions": [{
                "neutralSite": False,
                "status": {"type": {"completed": True}, "period": period},
                "competitors": [
                    {"homeAway": "home", "score": str(home_score),
                     "team": {"displayName": "Kansas City Chiefs"}},
                    {"homeAway": "away", "score": str(away_score),
                     "team": {"displayName": "Denver Broncos"}},
                ],
            }],
        }

    def test_regular_season_game_is_parsed(self):
        from pipeline.fetch_nfl import _parse_final_event

        parsed = _parse_final_event(self._event())
        assert parsed["home_team"] == "Chiefs"
        assert parsed["away_team"] == "Broncos"
        assert parsed["home_goals"] == 24
        assert parsed["week"] == 1

    def test_preseason_is_excluded(self):
        from pipeline.fetch_nfl import _parse_final_event

        assert _parse_final_event(self._event(season_type=1)) is None

    def test_overtime_is_flagged_and_score_is_final(self):
        from pipeline.fetch_nfl import _parse_final_event

        parsed = _parse_final_event(self._event(period=5, home_score=30, away_score=27))
        assert parsed["overtime"] is True
        assert parsed["home_goals"] == 30

    def test_regulation_tie_is_preserved(self):
        """NFL ties are legal and must survive as an equal scoreline."""
        from pipeline.fetch_nfl import _parse_final_event

        parsed = _parse_final_event(self._event(period=5, home_score=17, away_score=17))
        assert parsed["home_goals"] == parsed["away_goals"] == 17


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def _football_matches(teams, weeks=14, start="2026-09-06"):
    """Build a weekly round-robin schedule with a stable strength ordering."""
    base = datetime.strptime(start, "%Y-%m-%d")
    rows = []
    game_id = 0
    for week in range(weeks):
        date_str = (base + timedelta(days=7 * week)).strftime("%Y-%m-%d")
        for i in range(0, len(teams) - 1, 2):
            home = teams[(i + week) % len(teams)]
            away = teams[(i + week + 1) % len(teams)]
            # Earlier teams in the list are stronger.
            home_strong = teams.index(home) < teams.index(away)
            rows.append({
                "game_id": str(game_id),
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "home_goals": 28 if home_strong else 13,
                "away_goals": 13 if home_strong else 24,
                "neutral": False,
            })
            game_id += 1
    return pd.DataFrame(rows)


class TestNflPipelineEndToEnd:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_nfl_schedule")
    @patch("pipeline.run.fetch_nfl_games")
    def test_produces_valid_nfl_predictions(
        self, mock_games, mock_schedule, mock_odds, tmp_path, monkeypatch
    ):
        teams = ["Chiefs", "Bills", "Ravens", "Bengals", "Broncos", "Raiders"]
        mock_games.return_value = (_football_matches(teams), None)
        mock_schedule.return_value = [{
            "home_team": "Chiefs",
            "away_team": "Raiders",
            "date": _TODAY,
            "start_time": f"{_TODAY}T17:00:00Z",
            "completed": False,
            "neutral": False,
            "week": 15,
        }]
        mock_odds.return_value = [{
            "home_team": "Chiefs",
            "away_team": "Raiders",
            "commence_time": f"{_TODAY}T17:00:00Z",
            "home_odds": 1.45,
            "draw_odds": 0.0,
            "away_odds": 3.00,
        }]
        monkeypatch.setitem(SPORTS["nfl"], "results_feature_min_games", 10)

        output_dir = str(tmp_path / "nfl")
        run_sport_pipeline("nfl", output_dir=output_dir)

        with open(os.path.join(output_dir, "predictions.json")) as f:
            data = json.load(f)

        assert data["sport"] == "nfl"
        assert data["outcomes"] == ["home", "away"]
        assert len(data["matches"]) == 1

        match = data["matches"][0]
        assert set(match["model_probs"]) == {"home", "away"}
        assert abs(sum(match["model_probs"].values()) - 1.0) < 0.01
        assert "elo" in data["model_weights"]
        assert "results_features" in data["model_weights"]
        # Totals need a posted line; this fixture's odds carry none.
        assert data.get("totals_matches", []) == []
        assert data["full_slate"] is True
        assert data["diagnostics"]["fixtures_with_odds"] == 1

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_nfl_schedule")
    @patch("pipeline.run.fetch_nfl_games")
    def test_odds_are_requested_with_totals(
        self, mock_games, mock_schedule, mock_odds, tmp_path
    ):
        mock_games.return_value = (_football_matches(["Chiefs", "Bills"]), None)
        mock_schedule.return_value = []
        mock_odds.return_value = []

        run_sport_pipeline("nfl", output_dir=str(tmp_path / "nfl"))

        _, kwargs = mock_odds.call_args
        assert kwargs["sport_key"] == "americanfootball_nfl"
        assert kwargs["include_totals"] is True


class TestNcaafPipelineEndToEnd:
    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaaf_schedule")
    @patch("pipeline.run.fetch_ncaaf_games")
    def test_produces_valid_ncaaf_predictions(
        self, mock_games, mock_schedule, mock_odds, tmp_path, monkeypatch
    ):
        teams = ["Georgia", "Alabama", "Ohio State", "Texas", "Purdue", "Vanderbilt"]
        matches = _football_matches(teams)
        matches["conference_game"] = False
        mock_games.return_value = (matches, None)
        # A neutral-site non-conference game, which is a college staple.
        mock_schedule.return_value = [{
            "home_team": "Georgia",
            "away_team": "Purdue",
            "date": _TODAY,
            "start_time": f"{_TODAY}T20:00:00Z",
            "completed": False,
            "neutral": True,
            "conference_game": False,
            "week": 12,
        }]
        mock_odds.return_value = [{
            "home_team": "Georgia",
            "away_team": "Purdue",
            "commence_time": f"{_TODAY}T20:00:00Z",
            "home_odds": 1.30,
            "draw_odds": 0.0,
            "away_odds": 3.80,
        }]
        monkeypatch.setitem(SPORTS["ncaaf"], "results_feature_min_games", 10)

        output_dir = str(tmp_path / "ncaaf")
        run_sport_pipeline("ncaaf", output_dir=output_dir)

        with open(os.path.join(output_dir, "predictions.json")) as f:
            data = json.load(f)

        assert data["sport"] == "ncaaf"
        assert data["full_slate"] is True
        assert len(data["matches"]) == 1
        match = data["matches"][0]
        assert set(match["model_probs"]) == {"home", "away"}
        assert abs(sum(match["model_probs"].values()) - 1.0) < 0.01
        assert data["diagnostics"]["fixtures_with_odds"] == 1

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaaf_schedule")
    @patch("pipeline.run.fetch_ncaaf_games")
    def test_neutral_site_removes_home_advantage(
        self, mock_games, mock_schedule, mock_odds, tmp_path, monkeypatch
    ):
        teams = ["Georgia", "Alabama", "Ohio State", "Texas", "Purdue", "Vanderbilt"]
        mock_games.return_value = (_football_matches(teams), None)
        mock_odds.return_value = []
        monkeypatch.setitem(SPORTS["ncaaf"], "results_feature_min_games", 10)

        def _fixture(neutral):
            return [{
                "home_team": "Purdue",
                "away_team": "Georgia",
                "date": _TODAY,
                "start_time": f"{_TODAY}T20:00:00Z",
                "completed": False,
                "neutral": neutral,
            }]

        probs = {}
        for neutral in (False, True):
            mock_schedule.return_value = _fixture(neutral)
            output_dir = str(tmp_path / f"ncaaf_{neutral}")
            run_sport_pipeline("ncaaf", output_dir=output_dir)
            with open(os.path.join(output_dir, "predictions.json")) as f:
                data = json.load(f)
            probs[neutral] = data["matches"][0]["individual_models"]["elo"]["home"]

        # The weaker home team loses its home edge at a neutral site.
        assert probs[True] < probs[False]


class TestFullSlatePayload:
    """The slate table needs a pick and a probability on every row."""

    @pytest.mark.parametrize("sport_key", ["nfl", "ncaaf"])
    def test_full_slate_sports_enable_qualitative(self, sport_key):
        """Rows render the AI rationale, so the analysis layer must be on."""
        assert SPORTS[sport_key]["enable_qualitative"] is True

    @patch("pipeline.run.fetch_odds")
    @patch("pipeline.run.fetch_ncaaf_schedule")
    @patch("pipeline.run.fetch_ncaaf_games")
    def test_games_without_odds_still_get_a_pick(
        self, mock_games, mock_schedule, mock_odds, tmp_path, monkeypatch
    ):
        """College slates routinely outrun the books; those games still show."""
        teams = ["Georgia", "Alabama", "Ohio State", "Texas", "Purdue", "Vanderbilt"]
        mock_games.return_value = (_football_matches(teams), None)
        mock_schedule.return_value = [{
            "home_team": "Purdue",
            "away_team": "Georgia",
            "date": _TODAY,
            "start_time": f"{_TODAY}T20:00:00Z",
            "completed": False,
            "neutral": False,
        }]
        mock_odds.return_value = []          # no market for this fixture
        monkeypatch.setitem(SPORTS["ncaaf"], "results_feature_min_games", 10)

        output_dir = str(tmp_path / "ncaaf")
        run_sport_pipeline("ncaaf", output_dir=output_dir)

        with open(os.path.join(output_dir, "predictions.json")) as f:
            data = json.load(f)

        assert data["full_slate"] is True
        assert len(data["matches"]) == 1
        match = data["matches"][0]
        assert match["pick"] in {"home", "away"}
        assert match["model_prob"] is not None
        assert match["american_odds"] is None      # rendered as "--"
        assert data["slop_locks"] == []


class TestManifestActivation:
    def test_manifest_lists_football_as_active(self, tmp_path):
        from pipeline.run import _update_global_metadata

        for sport_key in SPORTS:
            sport_dir = tmp_path / sport_key
            sport_dir.mkdir()
            (sport_dir / "predictions.json").write_text(json.dumps({
                "generated_at": f"{_TODAY}T00:00:00Z",
                "diagnostics": {},
            }))

        _update_global_metadata(str(tmp_path))

        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        for sport_key, display in (("nfl", "NFL"), ("ncaaf", "NCAAF")):
            entry = manifest["sports"][sport_key]
            assert entry["status"] == "ok"
            assert entry["name"] == display
            assert entry.get("active") is not False
        # The season-disabled sport stays distinguishable.
        assert manifest["sports"]["ncaam"]["active"] is False
