from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pipeline.fetch_mma as fetch_mma
from pipeline.fetch_mma import fetch_mma_games, fetch_mma_schedule, normalize_mma_name


def _make_mma_scoreboard(completed: bool) -> dict:
    return {
        "events": [
            {
                "id": "600057366",
                "name": "UFC Fight Night: Adesanya vs. Pyfer",
                "date": "2026-03-28T21:00Z",
                "competitions": [
                    {
                        "id": "401863938",
                        "date": "2026-03-28T21:00Z",
                        "status": {
                            "type": {
                                "completed": completed,
                            }
                        },
                        "competitors": [
                            {
                                "order": 2,
                                "winner": False,
                                "athlete": {"displayName": "Bruna Brasil"},
                            },
                            {
                                "order": 1,
                                "winner": True,
                                "athlete": {"displayName": "Alexia Thainara"},
                            },
                        ],
                    }
                ],
            }
        ]
    }


class TestFetchMMASchedule:
    @patch("pipeline.fetch_mma.requests.get")
    def test_parses_upcoming_fight_from_athlete_payload(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = _make_mma_scoreboard(completed=False)
        mock_get.return_value = resp

        fixtures = fetch_mma_schedule()

        assert fixtures == [
            {
                "home_team": "Alexia Thainara",
                "away_team": "Bruna Brasil",
                "date": "2026-03-28",
                "start_time": "2026-03-28T21:00Z",
                "completed": False,
                "neutral": True,
            }
        ]

    @patch("pipeline.fetch_mma.requests.get")
    def test_includes_previous_et_day_in_scoreboard_window(self, mock_get, monkeypatch):
        class FixedDateTime:
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"events": []}
        mock_get.return_value = resp
        monkeypatch.setattr(fetch_mma, "datetime", FixedDateTime)

        fetch_mma_schedule()

        url = mock_get.call_args.args[0]
        assert "dates=20260328-20260412" in url


class TestFetchMMAGames:
    @patch("pipeline.fetch_mma.time.sleep")
    @patch("pipeline.fetch_mma.requests.get")
    def test_parses_completed_fight_from_athlete_payload(self, mock_get, mock_sleep):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = _make_mma_scoreboard(completed=True)
        mock_get.return_value = resp

        games_df, box_scores_df = fetch_mma_games(
            dates=["2026-03-28"],
            cache_path=None,
        )

        assert box_scores_df is None
        assert len(games_df) == 1
        row = games_df.iloc[0].to_dict()
        assert row["game_id"] == "401863938"
        assert row["date"] == "2026-03-28"
        assert row["home_team"] == "Alexia Thainara"
        assert row["away_team"] == "Bruna Brasil"
        assert row["home_goals"] == 1
        assert row["away_goals"] == 0
        mock_sleep.assert_called_once()


def test_normalize_mma_name_handles_aliases_and_accents():
    assert normalize_mma_name("Jiří Procházka") == "Jiri Prochazka"
    assert normalize_mma_name("Lando Vannata") == "Landon Vannata"
    assert normalize_mma_name("Charles Radtke") == "Charlie Radtke"
    assert normalize_mma_name("Loopy Godínez") == "Lupita Godinez"
    assert normalize_mma_name("Paulo Costa") == "Paulo Henrique Costa"
    assert normalize_mma_name("Abdul-Rakhman Yakhyaev") == "Abdulrakhman Yakhyaev"
    assert normalize_mma_name("Robert Ruchała") == "Robert Ruchala"
