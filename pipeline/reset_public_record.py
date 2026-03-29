"""Reset or prune the public pick record without touching model history.

Usage:
    python -m pipeline.reset_public_record
    python -m pipeline.reset_public_record --since 2026-03-28
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import DATA_DIR, RESULTS_LOG_FILENAME, SPORTS, TRACKING_DIRNAME
from pipeline.run import _compute_pick_stats


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _keep_row(row: dict, since: str | None, field_names: tuple[str, ...]) -> bool:
    if since is None:
        return False
    for field in field_names:
        value = str(row.get(field, ""))[:10]
        if value and value >= since:
            return True
    return False


def reset_public_record(data_dir: str | Path = DATA_DIR, since: str | None = None) -> None:
    """Clear or prune public pick-history stats.

    Parameters
    ----------
    data_dir:
        Base data directory containing per-sport folders.
    since:
        If provided, keep only picks on/after this ISO date (YYYY-MM-DD).
        If omitted, clear the public record entirely.
    """
    base_dir = Path(data_dir)
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for sport_key in SPORTS:
        sport_dir = base_dir / sport_key
        pick_history_path = sport_dir / "pick_history.json"
        predictions_path = sport_dir / "predictions.json"

        pick_history = _load_json(pick_history_path)
        picks = pick_history.get("picks", [])
        kept_picks = [
            pick for pick in picks
            if _keep_row(pick, since, ("pick_date", "match_date"))
        ]

        _save_json(pick_history_path, {
            "updated_at": updated_at,
            "picks": kept_picks,
        })

        predictions = _load_json(predictions_path)
        if predictions:
            predictions["pick_stats"] = _compute_pick_stats(kept_picks)
            predictions["generated_at"] = updated_at
            _save_json(predictions_path, predictions)

    results_log_path = base_dir / TRACKING_DIRNAME / RESULTS_LOG_FILENAME
    if results_log_path.exists():
        with results_log_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = [
                row for row in reader
                if _keep_row(row, since, ("match_date", "logged_at"))
            ]
            fieldnames = reader.fieldnames or []

        with results_log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset or prune the public pick record.")
    parser.add_argument(
        "--since",
        help="Keep only picks on/after this date (YYYY-MM-DD). Omit to clear all public record.",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Override the base data directory.",
    )
    args = parser.parse_args(argv)
    reset_public_record(args.data_dir, since=args.since)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
