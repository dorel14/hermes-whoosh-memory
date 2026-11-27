"""Outils agents supplémentaires pour whoosh-memory (spec section 5).

Outils :
- whoosh_open : lire un document / chunk par id ou path
- whoosh_forget : supprimer un document de l'index
- whoosh_stats : statistiques de l'index
- whoosh_add : ajouter un document manuellement à l'index
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .search import SearchHit, _Index, whoosh_search


def whoosh_open(
    index_dir: str | os.PathLike[str],
    *,
    id: Optional[str] = None,
    path: Optional[str] = None,
    offset: int = 0,
    max_chars: int = 4000,
) -> Optional[dict]:
    """Lit un document indexé par id ou path.

    Retourne le contenu stocké avec métadonnées, ou None si introuvable.
    """
    index_dir = Path(index_dir)
    index = _Index(index_dir)
    index.finish()

    if index._py is not None:
        for doc in index._py.docs:
            if (id and doc.get("id") == id) or (
                path and doc.get("path") == path
            ):
                text = doc.get("text", "")
                chunked = text[offset : offset + max_chars]
                return {
                    "id": doc.get("id"),
                    "path": doc.get("path"),
                    "title": doc.get("title"),
                    "content": chunked,
                    "offset": offset,
                    "max_chars": max_chars,
                    "total_chars": len(text),
                }
    return None


def whoosh_forget(
    index_dir: str | os.PathLike[str],
    *,
    id: Optional[str] = None,
    path: Optional[str] = None,
) -> bool:
    """Supprime un document de l'index (BM25 pur fallback)."""
    index_dir = Path(index_dir)
    index = _Index(index_dir)
    index.finish()

    if index._py is not None:
        before = len(index._py.docs)
        index._py.docs = [
            d
            for d in index._py.docs
            if not ((id and d.get("id") == id) or (path and d.get("path") == path))
        ]
        index._save_json()
        return len(index._py.docs) < before
    return False


def whoosh_stats(
    index_dir: str | os.PathLike[str],
) -> dict:
    """Statistiques de l'index whoosh-memory."""
    index_dir = Path(index_dir)
    index = _Index(index_dir)
    index.finish()

    n_docs = len(index._py.docs) if index._py is not None else 0
    types: dict[str, int] = {}
    if index._py is not None:
        for doc in index._py.docs:
            t = doc.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

    total_chars = sum(
        len(d.get("text", "")) for d in (index._py.docs if index._py is not None else [])
    )
    return {
        "index_dir": str(index_dir),
        "documents": n_docs,
        "by_type": types,
        "total_chars": total_chars,
        "backend": "whoosh" if index._whoosh is not None else "bm25_pure",
        "vector_available": index._whoosh is not None,
    }


def whoosh_add(
    index_dir: str | os.PathLike[str],
    *,
    id: str,
    path: str,
    title: str,
    content: str,
    type: str = "explicit",
    embedding: Optional[list[float]] = None,
) -> bool:
    """Ajoute un document manuellement à l'index."""
    if not content:
        return False
    index = _Index(Path(index_dir))
    index.add(doc_id=id, path=path, title=title, text=content, embedding=embedding)
    index.finish()
    return True


__all__ = ["whoosh_open", "whoosh_forget", "whoosh_stats", "whoosh_add"]
