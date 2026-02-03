"""
Lightweight JSONL telemetry logger for PokerVision.

The goal is to provide a single, safe function:

    log_event(path, event_dict)

which appends a JSON object per line to the given file path. The function:
  - creates parent directories if needed
  - never raises (all errors are swallowed after best‑effort logging)
  - relies only on the Python standard library
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory for `path` exists.

    This is kept private; callers should use `log_event` only.
    """
    parent = path.parent
    if not parent:
        return
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best‑effort only; failures are silently ignored.
        pass


def log_event(path: str, event: Dict[str, Any]) -> None:
    """
    Append a JSONL‑encoded event to the file at `path`.

    Behaviour:
    - Adds a UTC timestamp field `ts` if not already present.
    - Creates parent directories if required.
    - Opens the file in append mode and writes a single JSON line.
    - Swallows all exceptions so that telemetry can never break the main loop.
    """
    try:
        p = Path(path)
        _ensure_parent_dir(p)

        payload: Dict[str, Any] = dict(event) if event is not None else {}
        payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")

        # Use text mode with UTF‑8 for maximum portability.
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write(os.linesep)
    except Exception:
        # Telemetry should never raise; all errors are intentionally ignored.
        return

