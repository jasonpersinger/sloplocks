from typing import Optional, Union
"""Archive or explicitly rewrite the public pick record.

The default workflow is non-destructive. We export a pruned public-facing view
into ``tracking/public_record_archives`` and leave the canonical live files
alone. Rewriting the live record now requires an explicit opt-in flag because
the reporting surface is treated as operational history, not disposable copy.

Usage:
    python -m pipeline.reset_public_record
    python -m pipeline.reset_public_record --since 2026-03-28
    python -m pipeline.reset_public_record --since 2026-03-28 --rewrite-live-files
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import (
    DATA_DIR,
    PICK_DECISION_LOG_FILENAME,
    PUBLIC_RECORD_ARCHIVE_DIRNAME,
    RESULTS_AUDIT_LOG_FILENAME,
    RESULTS_LOG_FILENAME,
    SPORTS,
    TRACKING_DIRNAME,
)
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


def _keep_row(row: dict, since: Optional[str], field_names: tuple[str, ...]) -> bool:
    if since is None:
        return False
    for field in field_names:
        value = str(row.get(field, ""))[:10]
        if value and value >= since:
            return True
    return False


def _archive_dir(base_dir: Path, updated_at: str, since: Optional[str]) -> Path:
    stamp = updated_at.replace("-", "").replace(":", "").replace("T", "-").replace("Z", "")
    label = since or "empty"
    return base_dir / TRACKING_DIRNAME / PUBLIC_RECORD_ARCHIVE_DIRNAME / f"{stamp}-{label}"


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def reset_public_record(
    data_dir: Union[str, Path] = DATA_DIR,
    since: Optional[str] = None,
    rewrite_live_files: bool = False,
) -> Path:
    """Archive or explicitly rewrite public pick-history stats.

    Parameters
    ----------
    data_dir:
        Base data directory containing per-sport folders.
    since:
        If provided, keep only picks on/after this ISO date (YYYY-MM-DD).
        If omitted, the exported view is empty.
    rewrite_live_files:
        When ``False`` (default), only write an archived export. When ``True``,
        also rewrite the live public files after the archive is captured.
    """
    base_dir = Path(data_dir)
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    archive_dir = _archive_dir(base_dir, updated_at, since)

    archive_manifest = {
        "created_at": updated_at,
        "since": since,
        "rewrite_live_files": bool(rewrite_live_files),
        "sports": {},
        "results_log": None,
        "results_audit_log_preserved": str(base_dir / TRACKING_DIRNAME / RESULTS_AUDIT_LOG_FILENAME),
        "pick_decision_log_preserved": str(base_dir / TRACKING_DIRNAME / PICK_DECISION_LOG_FILENAME),
    }

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

        archived_pick_history = {
            "updated_at": updated_at,
            "source_path": str(pick_history_path),
            "picks": kept_picks,
        }
        archived_predictions = _load_json(predictions_path)
        if archived_predictions:
            archived_predictions["pick_stats"] = _compute_pick_stats(kept_picks)
            archived_predictions["generated_at"] = updated_at
            archived_predictions["source_path"] = str(predictions_path)

        archive_sport_dir = archive_dir / sport_key
        _save_json(archive_sport_dir / "pick_history.json", archived_pick_history)
        if archived_predictions:
            _save_json(archive_sport_dir / "predictions.json", archived_predictions)

        archive_manifest["sports"][sport_key] = {
            "pick_history_path": str(archive_sport_dir / "pick_history.json"),
            "predictions_path": str(archive_sport_dir / "predictions.json") if archived_predictions else None,
            "kept_picks": len(kept_picks),
        }

        if rewrite_live_files:
            # Fallback: live rewrites are still supported for emergency cleanup,
            # but only after an archive snapshot has been captured.
            _save_json(pick_history_path, archived_pick_history)
            if archived_predictions:
                _save_json(predictions_path, archived_predictions)

    results_log_path = base_dir / TRACKING_DIRNAME / RESULTS_LOG_FILENAME
    results_rows = []
    fieldnames: list[str] = []
    if results_log_path.exists():
        with results_log_path.open(newline="") as f:
            reader = csv.DictReader(f)
            results_rows = [
                row for row in reader
                if _keep_row(row, since, ("match_date", "logged_at"))
            ]
            fieldnames = list(reader.fieldnames or [])

    archive_results_path = archive_dir / RESULTS_LOG_FILENAME
    _write_rows(archive_results_path, fieldnames, results_rows)
    archive_manifest["results_log"] = {
        "path": str(archive_results_path),
        "rows": len(results_rows),
    }

    if rewrite_live_files and results_log_path.exists():
        _write_rows(results_log_path, fieldnames, results_rows)

    _save_json(archive_dir / "archive_manifest.json", archive_manifest)
    return archive_dir


def _main(argv:Optional[ list[str] ] = None) -> int:
    parser = argparse.ArgumentParser(description="Archive or explicitly rewrite the public pick record.")
    parser.add_argument(
        "--since",
        help="Keep only picks on/after this date (YYYY-MM-DD). Omit to export an empty public view.",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Override the base data directory.",
    )
    parser.add_argument(
        "--rewrite-live-files",
        action="store_true",
        help="Rewrite live public files after the archive snapshot is created.",
    )
    args = parser.parse_args(argv)
    reset_public_record(
        args.data_dir,
        since=args.since,
        rewrite_live_files=bool(args.rewrite_live_files),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
