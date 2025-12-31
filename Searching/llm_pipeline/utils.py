from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


def load_env() -> None:
    """
    Load environment variables from .env if present.
    Never prints secrets.
    """
    load_dotenv(override=False)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def sanitize_surrogates(s: str) -> str:
    """
    Remove invalid Unicode surrogate code points from a Python str.
    These can appear when parsing messy PDFs and will crash JSON UTF-8 encoding.
    """
    if not s:
        return s
    # Use ASCII placeholder to avoid Windows console encoding issues (e.g. GBK).
    return _SURROGATE_RE.sub("?", s)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize strings to make them safe for json.dumps + UTF-8 encoding.
    """
    if isinstance(obj, str):
        return sanitize_surrogates(obj)
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    return obj


def json_dumps(obj: Any) -> str:
    obj = sanitize_for_json(obj)
    return json.dumps(obj, ensure_ascii=False, indent=None, separators=(",", ":"))


def extract_last_json_object(text: str) -> dict[str, Any]:
    """
    Extract the last JSON object from a model response that may contain analysis text.
    Assumes the final block is a JSON object {...}.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty response text; cannot parse JSON.")

    # Greedy match for last {...} block. We intentionally do not support arrays here.
    matches = list(re.finditer(r"\{[\s\S]*\}\s*$", text.strip()))
    if not matches:
        # fallback: find any JSON-like object and pick the last
        matches = list(re.finditer(r"\{[\s\S]*\}", text))
    if not matches:
        raise ValueError("No JSON object found in response.")

    raw = matches[-1].group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw:\n{raw}") from e


@dataclass(frozen=True)
class SearchRecord:
    row_id: int
    filename: str
    title: str
    author_gs: str
    year_gs: Optional[float]
    venue_gs: str
    url_gs: str
    search_cluster: Optional[int]
    search_cluster_name: str
    source: str

    @staticmethod
    def from_pandas_row(r: Any) -> "SearchRecord":
        def get_str(key: str) -> str:
            v = r.get(key, "")
            return "" if v is None else sanitize_surrogates(str(v))

        def get_int_or_none(key: str) -> Optional[int]:
            v = r.get(key, None)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                return None
            try:
                return int(v)
            except Exception:
                return None

        def get_float_or_none(key: str) -> Optional[float]:
            v = r.get(key, None)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                return None
            try:
                return float(v)
            except Exception:
                return None

        row_id = r.get("row_id", None)
        try:
            row_id_int = int(row_id)
        except Exception:
            row_id_int = -1

        return SearchRecord(
            row_id=row_id_int,
            filename=get_str("filename"),
            title=get_str("title"),
            author_gs=get_str("author_gs"),
            year_gs=get_float_or_none("year_gs"),
            venue_gs=get_str("venue_gs"),
            url_gs=get_str("url_gs"),
            search_cluster=get_int_or_none("search_cluster"),
            search_cluster_name=get_str("search_cluster_name"),
            source=get_str("source"),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "filename": self.filename,
            "title": self.title,
            "author_gs": self.author_gs,
            "year_gs": self.year_gs,
            "venue_gs": self.venue_gs,
            "url_gs": self.url_gs,
            "search_cluster": self.search_cluster,
            "search_cluster_name": self.search_cluster_name,
            "source": self.source,
        }


