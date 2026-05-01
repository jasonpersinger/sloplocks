import csv
import json

from pipeline.reset_public_record import reset_public_record

ACTIVE_SPORTS = ("nba", "mlb", "nhl")


def _seed_public_record(data_dir):
    tracking_dir = data_dir / "tracking"
    tracking_dir.mkdir(parents=True)

    for sport in (*ACTIVE_SPORTS, "ncaam"):
        sport_dir = data_dir / sport
        sport_dir.mkdir(parents=True)
        with open(sport_dir / "pick_history.json", "w") as f:
            json.dump({
                "updated_at": "2026-03-29T00:00:00Z",
                "picks": [
                    {
                        "pick_date": "2026-03-27",
                        "match_date": "2026-03-27",
                        "type": "slop_lock",
                        "pick": "home",
                        "evaluated": True,
                        "won": True,
                        "decimal_odds": 2.0,
                    },
                    {
                        "pick_date": "2026-03-28",
                        "match_date": "2026-03-28",
                        "type": "slop_lock",
                        "pick": "home",
                        "evaluated": True,
                        "won": False,
                        "decimal_odds": 2.0,
                    },
                ],
            }, f)
        with open(sport_dir / "predictions.json", "w") as f:
            json.dump({
                "generated_at": "2026-03-29T00:00:00Z",
                "pick_stats": {"all": {"total": 999}},
            }, f)

    with open(tracking_dir / "results_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_date", "logged_at", "sport"])
        writer.writeheader()
        writer.writerow({"match_date": "2026-03-27", "logged_at": "2026-03-27T12:00:00Z", "sport": "nba"})
        writer.writerow({"match_date": "2026-03-28", "logged_at": "2026-03-28T12:00:00Z", "sport": "nba"})


def test_reset_public_record_archives_without_rewriting_live_files(tmp_path):
    data_dir = tmp_path / "data"
    _seed_public_record(data_dir)

    archive_dir = reset_public_record(data_dir, since="2026-03-28")

    for sport in ACTIVE_SPORTS:
        with open(data_dir / sport / "pick_history.json") as f:
            pick_history = json.load(f)
        assert len(pick_history["picks"]) == 2

        with open(archive_dir / sport / "pick_history.json") as f:
            archived = json.load(f)
        assert len(archived["picks"]) == 1
        assert archived["picks"][0]["pick_date"] == "2026-03-28"

        with open(archive_dir / sport / "predictions.json") as f:
            predictions = json.load(f)
        assert predictions["pick_stats"]["all"]["total"] == 1
        assert predictions["pick_stats"]["all"]["wins"] == 0
        assert predictions["pick_stats"]["all"]["losses"] == 1
    with open(data_dir / "ncaam" / "pick_history.json") as f:
        ncaam_history = json.load(f)
    assert len(ncaam_history["picks"]) == 2
    assert not (archive_dir / "ncaam" / "pick_history.json").exists()

    with open(data_dir / "tracking" / "results_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

    with open(archive_dir / "results_log.csv", newline="") as f:
        archived_rows = list(csv.DictReader(f))
    assert len(archived_rows) == 1
    assert archived_rows[0]["match_date"] == "2026-03-28"

    with open(archive_dir / "archive_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["results_audit_log_preserved"].endswith("results_audit_log.csv")
    assert manifest["pick_decision_log_preserved"].endswith("pick_decisions.csv")


def test_reset_public_record_can_explicitly_rewrite_live_files(tmp_path):
    data_dir = tmp_path / "data"
    _seed_public_record(data_dir)

    reset_public_record(data_dir, since="2026-03-28", rewrite_live_files=True)

    for sport in ACTIVE_SPORTS:
        with open(data_dir / sport / "pick_history.json") as f:
            pick_history = json.load(f)
        assert len(pick_history["picks"]) == 1
        assert pick_history["picks"][0]["pick_date"] == "2026-03-28"
    with open(data_dir / "ncaam" / "pick_history.json") as f:
        ncaam_history = json.load(f)
    assert len(ncaam_history["picks"]) == 2

    with open(data_dir / "tracking" / "results_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["match_date"] == "2026-03-28"
