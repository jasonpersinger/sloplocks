from __future__ import annotations
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

def _keep_row(row: dict, since: Optional[str], field_names: tuple[str, ...]) -> bool:
    """Return whether a CSV row should be kept based on date filter."""
    if not since:
        return True
    
    # Try to find a date field in the row
    match_date = None
    for key in ("match_date", "date", "logged_at"):
        if row.get(key):
            try:
                # Handle ISO timestamps or simple dates
                date_str = str(row[key])[:10]
                match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                break
            except (ValueError, TypeError):
                continue
                
    if not match_date:
        return True
        
    cutoff_date = datetime.strptime(since, "%Y-%m-%d").date()
    return match_date < cutoff_date


def _archive_dir(base_dir: Path, updated_at: str, since: Optional[str]) -> Path:
    """Return the path to the unique archive directory for this run."""
    since_label = f"before_{since}" if since else "full_reset"
    slug = f"reset_{updated_at}_{since_label}".replace(":", "").replace("-", "")
    return base_dir / TRACKING_DIRNAME / PUBLIC_RECORD_ARCHIVE_DIRNAME / slug


def prune_record(
    base_dir: Union[str, Path] = DATA_DIR,
    since: Optional[str] = None,
    rewrite_live: bool = False,
) -> dict:
    """Prune or archive the public tracking record.
    
    Parameters
    ----------
    base_dir : str or Path
        The project data directory.
    since : str or None
        If provided, only rows *before* this YYYY-MM-DD date are kept.
    rewrite_live : bool
        If True, actually overwrites the canonical log files. Use with caution.
    """
    base_dir = Path(base_dir)
    updated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = _archive_dir(base_dir, updated_at, since)
    
    files_to_prune = [
        RESULTS_LOG_FILENAME,
        RESULTS_AUDIT_LOG_FILENAME,
        PICK_DECISION_LOG_FILENAME,
    ]
    
    report = {
        "status": "success",
        "archive_dir": str(archive_path),
        "files": {},
        "rewrite_live": rewrite_live,
    }
    
    archive_path.mkdir(parents=True, exist_ok=True)
    
    for filename in files_to_prune:
        source_path = base_dir / TRACKING_DIRNAME / filename
        if not source_path.exists():
            continue
            
        with open(source_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_names = reader.fieldnames or ()
            all_rows = list(reader)
            
        kept_rows = [row for row in all_rows if _keep_row(row, since, field_names)]
        dropped_count = len(all_rows) - len(kept_rows)
        
        # Save archive
        with open(archive_path / filename, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(kept_rows)
            
        report["files"][filename] = {
            "total_rows": len(all_rows),
            "kept_rows": len(kept_rows),
            "dropped_rows": dropped_count,
        }
        
        if rewrite_live and dropped_count > 0:
            with open(source_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(kept_rows)
                
    # Also handle sport-specific history files if full reset
    if not since:
        for sport in SPORTS:
            history_path = base_dir / sport / "history.json"
            if history_path.exists():
                # Archive
                (archive_path / sport).mkdir(exist_ok=True)
                with open(history_path, "r") as f:
                    history_data = json.load(f)
                with open(archive_path / sport / "history.json", "w") as f:
                    json.dump(history_data, f, indent=2)
                
                if rewrite_live:
                    # Reset to empty structure
                    empty_history = {
                        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "predictions": []
                    }
                    with open(history_path, "w") as f:
                        json.dump(empty_history, f, indent=2)
                        
    return report


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prune or reset public pick records.")
    parser.add_argument("--since", help="Cutoff date (YYYY-MM-DD). Rows on/after this date are dropped.")
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
    sys.exit(_main())
