"""
Quick telemetry inspector for PokerVision.

Usage:
    python scripts/inspect_telemetry.py

It reads `logs/telemetry.jsonl` (if present) and prints:
  - total number of events
  - number of unique opponents
  - top 5 opponents by event count
  - one pretty‑printed sample event
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole script.
                continue
    return events


def main() -> None:
    telemetry_path = Path("logs/telemetry.jsonl")
    events = load_events(telemetry_path)

    if not events:
        print("No telemetry events found at", telemetry_path)
        return

    # Basic counts
    total_events = len(events)
    opponents = [str(e.get("opponent") or "_unknown") for e in events]
    unique_opponents = set(opponents)

    print("Telemetry summary")
    print("=================")
    print(f"File:           {telemetry_path}")
    print(f"Total events:   {total_events}")
    print(f"Unique opponents: {len(unique_opponents)}")
    print()

    # Top 5 opponents by event count
    counter = Counter(opponents)
    print("Top 5 opponents by event count:")
    for name, count in counter.most_common(5):
        label = "(unknown)" if name == "_unknown" else name
        print(f"  {label:20s} {count:5d}")

    print()
    print("Sample event")
    print("============")
    sample = events[-1]  # most recent event
    print(json.dumps(sample, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

