"""cache_manager.py
Persistent disk cache for the Parallel Battery Analytics app.

Strategy
--------
* Every cacheable object gets a **fingerprint** (MD5 of content or file
  mtimes).  If the fingerprint matches what is stored on disk the cached
  version is loaded; otherwise the object is recomputed and the cache is
  updated.

Supported objects
-----------------
| Key              | What is cached            | Format         |
|------------------|---------------------------|----------------|
| dataset_bundle   | raw loaded bundle         | joblib pickle  |
| prepared_data    | PreparedData dataclass    | joblib pickle  |
| feature_df       | engineered feature table  | parquet        |
| model_<name>     | trained ModelingResult    | joblib pickle  |
| soh_hist         | per-cycle SOH table       | parquet        |

Cache directory
---------------
Defaults to  ``./battery_cache/`` next to app.py.
Override with env-var  ``BATTERY_CACHE_DIR``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR = Path(os.environ.get("BATTERY_CACHE_DIR", "battery_cache"))
_META_FILE = "cache_meta.json"
_VERSION = "v3"          # bump to invalidate all existing caches


def get_cache_dir() -> Path:
    d = _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------

def _hash_str(s: str) -> str:
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:16]


def _hash_df(df: pd.DataFrame) -> str:
    """Fast hash of a DataFrame using shape + dtypes + sample values."""
    sig = f"{df.shape}|{list(df.dtypes)}|{df.head(5).to_csv()}|{df.tail(5).to_csv()}"
    return _hash_str(sig)


def fingerprint_path(path: str) -> str:
    """
    Hash a dataset path: combines the path string with the modification
    times of all files under it (up to 200 files for speed).
    """
    p = Path(path)
    parts = [path]
    try:
        if p.is_file():
            parts.append(str(p.stat().st_mtime))
        elif p.is_dir():
            files = sorted(p.rglob("*"))[:200]
            for f in files:
                if f.is_file():
                    parts.append(f"{f}:{f.stat().st_mtime:.0f}")
    except Exception:
        pass
    return _hash_str("|".join(parts))


def fingerprint_dfs(*dfs: pd.DataFrame) -> str:
    """Combined fingerprint of one or more DataFrames."""
    return _hash_str("|".join(_hash_df(df) for df in dfs if df is not None))


# ---------------------------------------------------------------------------
# Low-level read / write
# ---------------------------------------------------------------------------

def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / _META_FILE


def _load_meta(cache_dir: Path) -> dict:
    mp = _meta_path(cache_dir)
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            pass
    return {}


def _save_meta(cache_dir: Path, meta: dict) -> None:
    try:
        _meta_path(cache_dir).write_text(json.dumps(meta, indent=2))
    except Exception as exc:
        log.warning("Could not write cache meta: %s", exc)


def _joblib_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.joblib"


def _parquet_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.parquet"


def _write_object(cache_dir: Path, key: str, obj: Any, use_parquet: bool = False) -> bool:
    try:
        if use_parquet and isinstance(obj, pd.DataFrame):
            p = _parquet_path(cache_dir, key)
            obj.to_parquet(p, index=False, engine="pyarrow", compression="snappy")
        else:
            p = _joblib_path(cache_dir, key)
            joblib.dump(obj, p, compress=("lz4", 3))
        return True
    except Exception as exc:
        log.warning("Cache write failed for key=%s: %s", key, exc)
        return False


def _read_object(cache_dir: Path, key: str, use_parquet: bool = False) -> Any:
    try:
        if use_parquet:
            p = _parquet_path(cache_dir, key)
            if p.exists():
                return pd.read_parquet(p, engine="pyarrow")
        p = _joblib_path(cache_dir, key)
        if p.exists():
            return joblib.load(p)
    except Exception as exc:
        log.warning("Cache read failed for key=%s: %s", key, exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_set(
    key: str,
    obj: Any,
    fingerprint: str,
    use_parquet: bool = False,
    cache_dir: Optional[Path] = None,
) -> None:
    """Save *obj* to disk under *key* with a *fingerprint* tag."""
    d = cache_dir or get_cache_dir()
    ok = _write_object(d, key, obj, use_parquet=use_parquet)
    if ok:
        meta = _load_meta(d)
        meta[key] = {
            "fingerprint": fingerprint,
            "version":     _VERSION,
            "saved_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "parquet":     use_parquet,
        }
        _save_meta(d, meta)
        log.info("Cache SET  key=%-30s  fp=%s", key, fingerprint)


def cache_get(
    key: str,
    fingerprint: str,
    cache_dir: Optional[Path] = None,
) -> Optional[Any]:
    """
    Load cached object for *key* if the stored fingerprint matches.
    Returns ``None`` on cache-miss, stale data, or any error.
    """
    d = cache_dir or get_cache_dir()
    meta = _load_meta(d)
    entry = meta.get(key, {})
    if entry.get("fingerprint") != fingerprint or entry.get("version") != _VERSION:
        log.debug("Cache MISS key=%s", key)
        return None
    obj = _read_object(d, key, use_parquet=entry.get("parquet", False))
    if obj is not None:
        log.info("Cache HIT  key=%-30s  fp=%s", key, fingerprint)
    return obj


def cache_delete(key: str, cache_dir: Optional[Path] = None) -> None:
    """Remove a single cached entry."""
    d = cache_dir or get_cache_dir()
    for p in [_joblib_path(d, key), _parquet_path(d, key)]:
        if p.exists():
            p.unlink(missing_ok=True)
    meta = _load_meta(d)
    meta.pop(key, None)
    _save_meta(d, meta)


def cache_clear_all(cache_dir: Optional[Path] = None) -> int:
    """Delete all cached files. Returns number of files removed."""
    d = cache_dir or get_cache_dir()
    count = 0
    for f in d.glob("*.joblib"):
        f.unlink(missing_ok=True); count += 1
    for f in d.glob("*.parquet"):
        f.unlink(missing_ok=True); count += 1
    if _meta_path(d).exists():
        _meta_path(d).unlink(missing_ok=True); count += 1
    log.info("Cache CLEAR  %d files removed", count)
    return count


def cache_list(cache_dir: Optional[Path] = None) -> list[dict]:
    """Return list of metadata dicts for all cached entries."""
    d = cache_dir or get_cache_dir()
    meta = _load_meta(d)
    rows = []
    for key, info in meta.items():
        fp = _joblib_path(d, key) if not info.get("parquet") else _parquet_path(d, key)
        size_kb = round(fp.stat().st_size / 1024, 1) if fp.exists() else 0
        rows.append({
            "key":       key,
            "saved_at":  info.get("saved_at", ""),
            "size_kb":   size_kb,
            "version":   info.get("version", ""),
        })
    return sorted(rows, key=lambda r: r["saved_at"], reverse=True)


def cache_size_mb(cache_dir: Optional[Path] = None) -> float:
    d = cache_dir or get_cache_dir()
    return round(sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1024 / 1024, 2)
