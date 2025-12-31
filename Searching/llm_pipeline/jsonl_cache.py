from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    def _gen():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    return _gen()


def load_result_map_by_filename(path: str | Path) -> dict[str, dict[str, Any]]:
    """
    filename -> parsed_result (obj["result"]) for rows that have dict result.
    If a filename appears multiple times, the last one wins.
    """
    m: dict[str, dict[str, Any]] = {}
    for obj in iter_jsonl(path):
        fn = obj.get("filename")
        res = obj.get("result")
        if isinstance(fn, str) and fn and isinstance(res, dict):
            m[fn] = res
    return m


def load_processed_filenames(path: str | Path) -> set[str]:
    """
    Consider a filename "processed" if it appears in the jsonl at least once,
    regardless of success/failure. This is used for resume/skip logic.
    """
    s: set[str] = set()
    for obj in iter_jsonl(path):
        fn = obj.get("filename")
        if isinstance(fn, str) and fn:
            s.add(fn)
    return s



