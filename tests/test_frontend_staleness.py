"""Static frontend checks for stale-pick suppression."""

import re
import subprocess
from pathlib import Path


def test_frontend_freshness_helper_rejects_started_picks():
    source = Path("index.html").read_text()
    match = re.search(
        r"function isFreshPick\(pick, nowMs\) \{(?P<body>.*?)\n  \}\n\n  function filterFreshPicks",
        source,
        re.S,
    )
    assert match, "index.html should expose a pure isFreshPick helper"

    script = (
        "function isFreshPick(pick, nowMs) {"
        + match.group("body")
        + "\n}\n"
        + """
const now = Date.parse('2026-06-10T16:00:00Z');
if (isFreshPick({ start_time: '2026-06-10T15:59:59Z' }, now)) process.exit(1);
if (!isFreshPick({ start_time: '2026-06-10T16:00:01Z' }, now)) process.exit(2);
if (isFreshPick({ date: '2026-06-09' }, now)) process.exit(3);
if (!isFreshPick({ date: '2026-06-10' }, now)) process.exit(4);
"""
    )

    subprocess.run(["node", "-e", script], check=True)


def test_frontend_filters_picks_before_rendering():
    source = Path("index.html").read_text()

    assert "renderFeed(filterFreshPicks(allPicks));" in source
    assert "renderFeed(filterFreshPicks(locks));" in source
