"""Archive or explicitly rewrite the public pick record.

The default workflow is non-destructive. We export a pruned public-facing view
into ``tracking/public_record_archives`` and leave the canonical live files
alone. Rewriting the live record requires an explicit opt-in flag because the
reporting surface is treated as operational history, not disposable copy.

Usage:
    python -m pipeline.reset_public_record
    python -m pipeline.reset_public_record --since 2026-03-28
    python -m pipeline.reset_public_record --since 2026-03-28 --rewrite-live-files
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

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
    with path.open() as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _keep_row(row: dict, since: Optional[str], field_names: tuple[str, ...]) -> bool:
    """Return whether a row belongs in the pruned public-facing record."""
    if since is None:
        return False

    cutoff_date = datetime.strptime(since, "%Y-%m-%d").date()
    for field_name in field_names:
        value = str(row.get(field_name, ""))[:10]
        if not value:
            continue
        try:
            row_date = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            continue
        if row_date >= cutoff_date:
            return True
    return False


def _archive_dir(base_dir: Path, updated_at: str, since: Optional[str]) -> Path:
    since_label = f"since_{since}" if since else "full_reset"
    slug = f"reset_{updated_at}_{since_label}".replace(":", "").replace("-", "")
    return base_dir / TRACKING_DIRNAME / PUBLIC_RECORD_ARCHIVE_DIRNAME / slug


def prune_record(
    base_dir: Union[str, Path] = DATA_DIR,
    since: Optional[str] = None,
    rewrite_live: bool = False,
) -> dict:
    """Archive a pruned public view and optionally rewrite live files."""
    base_dir = Path(base_dir)
    updated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = _archive_dir(base_dir, updated_at, since)
    archive_path.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "success",
        "archive_dir": str(archive_path),
        "files": {},
        "rewrite_live": rewrite_live,
    }

    for sport_key in SPORTS:
        sport_dir = base_dir / sport_key
        pick_history_path = sport_dir / "pick_history.json"
        predictions_path = sport_dir / "predictions.json"

        if not pick_history_path.exists() and not predictions_path.exists():
            continue

        pick_history = _load_json(pick_history_path)
        picks = list(pick_history.get("picks", []))
        kept_picks = [
            pick for pick in picks
            if _keep_row(pick, since, ("pick_date", "match_date"))
        ]

        archived_history = {
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "picks": kept_picks,
        }
        _save_json(archive_path / sport_key / "pick_history.json", archived_history)

        predictions = _load_json(predictions_path)
        if predictions:
            archived_predictions = dict(predictions)
            archived_predictions["generated_at"] = archived_history["updated_at"]
            archived_predictions["pick_stats"] = _compute_pick_stats(kept_picks)
            _save_json(archive_path / sport_key / "predictions.json", archived_predictions)

        report["files"][sport_key] = {
            "total_picks": len(picks),
            "archived_picks": len(kept_picks),
        }

        if rewrite_live:
            _save_json(pick_history_path, archived_history)
            if predictions:
                _save_json(predictions_path, archived_predictions)

    for filename in (
        RESULTS_LOG_FILENAME,
        RESULTS_AUDIT_LOG_FILENAME,
        PICK_DECISION_LOG_FILENAME,
    ):
        source_path = base_dir / TRACKING_DIRNAME / filename
        if not source_path.exists():
            continue

        with source_path.open(mode="r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            field_names = reader.fieldnames or []
            all_rows = list(reader)

        kept_rows = [
            row for row in all_rows
            if _keep_row(row, since, tuple(field_names))
        ]

        with (archive_path / filename).open(mode="w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names)
            if field_names:
                writer.writeheader()
                writer.writerows(kept_rows)

        report["files"][filename] = {
            "total_rows": len(all_rows),
            "archived_rows": len(kept_rows),
        }

        if rewrite_live:
            with source_path.open(mode="w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=field_names)
                if field_names:
                    writer.writeheader()
                    writer.writerows(kept_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "since": since,
        "rewrite_live": rewrite_live,
        "results_audit_log_preserved": str(base_dir / TRACKING_DIRNAME / RESULTS_AUDIT_LOG_FILENAME),
        "pick_decision_log_preserved": str(base_dir / TRACKING_DIRNAME / PICK_DECISION_LOG_FILENAME),
    }
    _save_json(archive_path / "archive_manifest.json", manifest)
    report["archive_manifest"] = str(archive_path / "archive_manifest.json")

    return report


def reset_public_record(
    data_dir: Union[str, Path] = DATA_DIR,
    since: Optional[str] = None,
    rewrite_live_files: bool = False,
) -> Path:
    """Compatibility wrapper returning the archive directory path."""
    report = prune_record(
        base_dir=data_dir,
        since=since,
        rewrite_live=rewrite_live_files,
    )
    return Path(report["archive_dir"])


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prune or reset public pick records.")
    parser.add_argument("--since", help="Keep only picks on/after this date (YYYY-MM-DD).")
    parser.add_argument("--rewrite-live-files", action="store_true", help="Actually overwrite live files.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Data directory path.")

    args = parser.parse_args(argv)

    try:
        report = prune_record(
            base_dir=args.data_dir,
            since=args.since,
            rewrite_live=args.rewrite_live_files,
        )
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(f"Error resetting records: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
