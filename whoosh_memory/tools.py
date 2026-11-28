"""Fonctions d'accès à l'index whoosh-memory : open, forget, stats, add.

Elles fonctionnent avec les deux moteurs de ``search._Index`` : l'index
whoosh-ng (cas normal) et le repli BM25 pur Python (whoosh-ng absent).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .search import _Index


def whoosh_open(
    index_dir: str | os.PathLike[str],
    *,
    id: Optional[str] = None,
    path: Optional[str] = None,
    offset: int = 0,
    max_chars: int = 4000,
) -> Optional[dict]:
    """Lit un document indexé par id ou path (contenu paginé), ou None."""
    doc = _Index(Path(index_dir)).get(doc_id=id, path=path)
    if doc is None:
        return None
    text = doc.pop("text", "") or ""
    offset = max(0, int(offset))
    max_chars = max(1, int(max_chars))
    return {
        **doc,
        "content": text[offset : offset + max_chars],
        "offset": offset,
        "max_chars": max_chars,
        "total_chars": len(text),
    }


def whoosh_forget(
    index_dir: str | os.PathLike[str],
    *,
    id: Optional[str] = None,
    path: Optional[str] = None,
) -> int:
    """Supprime les documents correspondant à id ou path ; renvoie le nombre supprimé."""
    index = _Index(Path(index_dir))
    removed = index.delete(doc_id=id, path=path)
    index.finish()
    return removed


def whoosh_stats(index_dir: str | os.PathLike[str]) -> dict:
    """Statistiques de l'index whoosh-memory."""
    stats = _Index(Path(index_dir)).stats()
    return {"index_dir": str(index_dir), **stats}


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
    """Ajoute ou remplace (même ``id``) un document dans l'index."""
    if not content:
        return False
    index = _Index(Path(index_dir))
    index.add(doc_id=id, path=path, title=title, text=content, embedding=embedding, doc_type=type)
    index.finish()
    return True


__all__ = ["whoosh_open", "whoosh_forget", "whoosh_stats", "whoosh_add"]
