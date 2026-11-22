"""Configuration loader for the whoosh-memory Hermes plugin.

Reads ``$HERMES_HOME/plugin-data/whoosh-memory/config.json`` and merges it
over the defaults below (spec section 8). 100% local, no network calls, no
secrets stored — this module only loads plain JSON and expands ``~`` in path
listings. A missing or invalid config file never crashes the plugin: it falls
back to ``DEFAULTS``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = ["DEFAULTS", "hermes_home", "CONFIG_PATH", "load_config"]

#: Default configuration — mirrors spec section 8 ("Configuration exemple").
DEFAULTS: Dict[str, Any] = {
    # --- Indexing sources (P0/P1) ---
    "index_sessions": True,
    "index_tool_results": False,
    "index_paths": ["~/Documents", "~/notes"],
    "index_skills": True,
    "exclude_patterns": ["*.log", "node_modules", ".git"],
    # --- Chunking ---
    "chunk_size": 1000,
    # --- Prefetch ---
    "prefetch_top_k": 3,
    "prefetch_max_chars": 3000,
    "prefetch_min_score": 0.25,
    "prefetch_use_hybrid": True,
    # --- Hybrid search (Phase 2; BM25-only is the MVP fallback) ---
    "hybrid_enabled": "auto",
    "hybrid_method": "rrf",
    "hybrid_weight_bm25": 0.65,
    "hybrid_weight_vector": 0.35,
    # --- Embeddings (local, optional) ---
    "embedding_provider": "fastembed",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_dim": 384,
    "embedding_cache": True,
    # --- File indexing limits / lifecycle ---
    "max_file_mb": 8,
    "background_refresh": True,
}


def hermes_home() -> Path:
    """Resolve ``$HERMES_HOME`` (fallback: /opt/data)."""
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser()
    return Path("/opt/data")


#: Absolute path to the plugin's config file under HERMES_HOME.
CONFIG_PATH: Path = hermes_home() / "plugin-data" / "whoosh-memory" / "config.json"


def _expand_paths(value: Any) -> Any:
    """Expand ``~`` in a string or in every entry of a list of strings."""
    if isinstance(value, str):
        return os.path.expanduser(value)
    if isinstance(value, list):
        return [os.path.expanduser(v) if isinstance(v, str) else v for v in value]
    return value


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load config, merging the user file over ``DEFAULTS``.

    Args:
        path: Optional override for the config file location. Defaults to
            ``$HERMES_HOME/plugin-data/whoosh-memory/config.json``.

    Returns:
        A merged configuration dict. ``~`` is expanded in ``index_paths``.
        Never raises on a missing or malformed file — the defaults win.
    """
    cfg: Dict[str, Any] = dict(DEFAULTS)
    p = Path(path) if path is not None else CONFIG_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return cfg
    except (json.JSONDecodeError, OSError, ValueError):
        # Invalid config must never break the plugin; fall back to defaults.
        return cfg

    if isinstance(data, dict):
        # Keep unknown keys for forward compatibility; drop explicit nulls.
        cfg.update({k: v for k, v in data.items() if v is not None})

    if isinstance(cfg.get("index_paths"), list):
        cfg["index_paths"] = _expand_paths(cfg["index_paths"])
    return cfg


#: Convenience handle — the resolved config at import time (reload via load_config()).
config: Dict[str, Any] = load_config()
