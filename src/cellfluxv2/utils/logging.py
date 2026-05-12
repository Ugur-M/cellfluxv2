"""Tiny JSONL append/read helpers + a metrics formatter for stdout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single JSON object (one line, sort_keys=True, flushed)."""
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict; got {type(record).__name__}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(record, f, sort_keys=True)
        f.write("\n")
        f.flush()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Blank lines are skipped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"jsonl not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_metrics(
    metrics: dict[str, float | int], prefix: str | None = None
) -> str:
    """Format a metrics dict into a deterministic ``key=value`` string.

    Keys are sorted alphabetically. Floats outside ``[1e-3, 1e6)`` use
    scientific notation; everything else uses ``%.4f``. Integers are
    formatted with ``str`` (no decimals).
    """
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics must be a dict; got {type(metrics).__name__}")
    parts: list[str] = []
    if prefix:
        parts.append(str(prefix))
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, bool) or isinstance(v, int):
            parts.append(f"{k}={v}")
        elif isinstance(v, float):
            absv = abs(v)
            if v == 0.0 or (1e-3 <= absv < 1e6):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v:.3e}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)
