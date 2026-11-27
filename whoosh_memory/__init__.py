from __future__ import annotations

from dataclasses import asdict
from typing import Optional

"""Package whoosh-memory (plugin Hermes).

Fournit un backend de mémoire local hybride (BM25 + embeddings) indexant
l'historique des sessions, MEMORY.md / USER.md, les documents locaux, les
skills Hermes et les images dans un index whoosh-ng pour une récupération
hors-ligne rapide avec prefetch automatique avant chaque turn.

Le plugin est 100 % local : aucun appel réseau n'est nécessaire. La recherche
vectorielle utilise le champ natif ``whoosh_modern.fields.VECTOR`` (v0.3.0).
"""

from .config import DEFAULTS, load_config
from .images import (
    IMAGE_EXTENSIONS,
    image_metadata,
    image_to_base64,
    index_images,
    search_images,
    whoosh_image_base64,
    whoosh_image_info,
)
from .search import SearchHit, index_documents, whoosh_search
from .tools import whoosh_add, whoosh_forget, whoosh_open, whoosh_stats

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "register",
    "DEFAULTS",
    "load_config",
    "SearchHit",
    "whoosh_search",
    "whoosh_open",
    "whoosh_forget",
    "whoosh_stats",
    "whoosh_add",
    "index_documents",
    "index_images",
    "search_images",
    "whoosh_image_base64",
    "whoosh_image_info",
    "image_metadata",
    "image_to_base64",
    "IMAGE_EXTENSIONS",
]

def register(ctx) -> None:
    """Point d'entrée du plugin (appelé par Hermes au chargement).

    Enregistre les outils, hooks et commandes fournis par whoosh-memory.

    Args:
        ctx: contexte du plugin fourni par Hermes (outils, hooks, log, config...).
    """
    log = getattr(ctx, "log", None) or (lambda *a, **k: None)

    # --- Outils agents ---
    tools = {
        "whoosh_search": _tool_whoosh_search,
        "whoosh_open": _tool_whoosh_open,
        "whoosh_forget": _tool_whoosh_forget,
        "whoosh_stats": _tool_whoosh_stats,
        "whoosh_add": _tool_whoosh_add,
        "whoosh_image_base64": _tool_whoosh_image_base64,
        "whoosh_image_info": _tool_whoosh_image_info,
    }

    register_tool = getattr(ctx, "register_tool", None)
    if register_tool is not None:
        for name, func in tools.items():
            try:
                register_tool(name, func)
                if log:
                    log(f"whoosh-memory: outil enregistré {name}")
            except Exception as exc:  # pragma: no cover - défense
                if log:
                    log(f"whoosh-memory: échec enregistrement {name}: {exc}")
    else:
        # Fallback : exposer les outils directement sur le contexte.
        for name, func in tools.items():
            setattr(ctx, name, func)

    # --- Hooks ---
    hooks = {
        "prefetch": _hook_prefetch,
        "sync_turn": _hook_sync_turn,
        "on_memory_write": _hook_on_memory_write,
        "on_session_end": _hook_on_session_end,
        "on_pre_compress": _hook_on_pre_compress,
    }

    register_hook = getattr(ctx, "register_hook", None)
    if register_hook is not None:
        for name, func in hooks.items():
            try:
                register_hook(name, func)
                if log:
                    log(f"whoosh-memory: hook enregistré {name}")
            except Exception as exc:  # pragma: no cover - défense
                if log:
                    log(f"whoosh-memory: échec hook {name}: {exc}")

    log("whoosh-memory v0.3.0 enregistré")


# ---------------------------------------------------------------------------
# Adaptateurs d'outils (signature Hermes : kwargs depuis les arguments agent)
# ---------------------------------------------------------------------------

def _get_index_dir(ctx) -> str:
    cfg = getattr(ctx, "config", None) or {}
    return str(
        cfg.get(
            "index_dir",
            str(__import__("pathlib").Path(
                __import__("os").environ.get("HERMES_HOME", "/opt/data")
            ) / "plugin-data" / "whoosh-memory" / "index"),
        )
    )


def _tool_whoosh_search(ctx, **kwargs) -> list[dict]:
    from .config import load_config

    cfg = load_config()
    index_dir = kwargs.get("index_dir") or _get_index_dir(ctx)
    hits = whoosh_search(
        index_dir,
        kwargs.get("query", ""),
        top_k=int(kwargs.get("limit", 10)),
        hybrid=kwargs.get("hybrid", "auto"),
    )
    return [asdict(h) for h in hits]


def _tool_whoosh_open(ctx, **kwargs) -> Optional[dict]:
    index_dir = kwargs.get("index_dir") or _get_index_dir(ctx)
    return whoosh_open(
        index_dir,
        id=kwargs.get("id"),
        path=kwargs.get("path"),
        offset=int(kwargs.get("offset", 0)),
        max_chars=int(kwargs.get("max_chars", 4000)),
    )


def _tool_whoosh_forget(ctx, **kwargs) -> bool:
    index_dir = kwargs.get("index_dir") or _get_index_dir(ctx)
    return whoosh_forget(
        index_dir,
        id=kwargs.get("id"),
        path=kwargs.get("path"),
    )


def _tool_whoosh_stats(ctx, **kwargs) -> dict:
    index_dir = kwargs.get("index_dir") or _get_index_dir(ctx)
    return whoosh_stats(index_dir)


def _tool_whoosh_add(ctx, **kwargs) -> bool:
    index_dir = kwargs.get("index_dir") or _get_index_dir(ctx)
    return whoosh_add(
        index_dir,
        id=kwargs["id"],
        path=kwargs["path"],
        title=kwargs.get("title", ""),
        content=kwargs.get("content", ""),
        type=kwargs.get("type", "explicit"),
    )


def _tool_whoosh_image_base64(ctx, **kwargs) -> Optional[dict]:
    return whoosh_image_base64(
        kwargs.get("path", ""),
        max_size_mb=float(kwargs.get("max_size_mb", 10)),
    )


def _tool_whoosh_image_info(ctx, **kwargs) -> Optional[dict]:
    index_dir = kwargs.get("index_dir") or _get_index_dir(ctx)
    return whoosh_image_info(index_dir, kwargs.get("path", "") or kwargs.get("id", ""))


# ---------------------------------------------------------------------------
# Hooks (stubs fonctionnels, à brancher sur l'historique Hermes)
# ---------------------------------------------------------------------------

def _hook_prefetch(ctx, **kwargs) -> list[dict]:
    """Prefetch automatique : retourne les top-k documents pertinents avant un turn."""
    index_dir = _get_index_dir(ctx)
    query = kwargs.get("query", "")
    if not query:
        return []
    hits = whoosh_search(index_dir, query, top_k=int(kwargs.get("top_k", 3)))
    return [asdict(h) for h in hits]


def _hook_sync_turn(ctx, **kwargs) -> None:
    """Indexe un tour de conversation (user + assistant) après chaque turn."""
    # À brancher sur l'historique Hermes : indexer user_message + assistant_message.
    pass


def _hook_on_memory_write(ctx, **kwargs) -> None:
    """Mirror immédiat de MEMORY.md / USER.md dans l'index."""
    pass


def _hook_on_session_end(ctx, **kwargs) -> None:
    """Nettoyage / finalisation à la fin d'une session."""
    pass


def _hook_on_pre_compress(ctx, **kwargs) -> None:
    """Sauvegarde de l'index avant compression du contexte."""
    pass
