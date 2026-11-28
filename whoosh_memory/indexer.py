"""Indexation incrémentale whoosh-ng (spec Section 4).

- Index principal  : $HERMES_HOME/plugin-data/whoosh-memory/index/
- Index skills     : $HERMES_HOME/plugin-data/whoosh-memory/skills-index/
- content_hash     : SHA-256(contenu [+ mtime + taille]) pour détecter les changements
- Writer whoosh-ng : utilisé en context manager (commit automatique à la sortie)
- Corruption       : l'index est recréé automatiquement s'il est illisible/absent
- Réseau           : aucun appel réseau ; pure Python uniquement
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

try:
    from whoosh.index import Index as WhooshIndex
    from whoosh.index import create_in, open_dir
except ImportError as exc:  # whoosh-ng est une dépendance déclarée du plugin
    raise ImportError(
        "whoosh-ng est requis par le plugin whoosh-memory. "
        "Installez-le (ex. `pip install whoosh-ng`) avant d'importer ce module."
    ) from exc

from .schema import TYPE_SESSION_TURN, SCHEMA, get_schema

logger = logging.getLogger("hermes.plugins.whoosh-memory.indexer")

# No module-level hardcoded HERMES_HOME/INDEX_DIR — index paths are passed in
# explicitly by the caller (provider.initialize resolves them from hermes_home).


def content_hash(content: str, mtime: Optional[float] = None,
                 size: Optional[int] = None) -> str:
    """SHA-256 du contenu (augmenté de mtime/taille quand fournis) pour détection de changement."""
    digest = hashlib.sha256(content.encode("utf-8"))
    if mtime is not None:
        digest.update(b"|mtime=" + repr(mtime).encode("ascii"))
    if size is not None:
        digest.update(b"|size=" + repr(size).encode("ascii"))
    return digest.hexdigest()


def _open_or_recreate(index_dir: str, schema) -> WhooshIndex:
    """Ouvre l'index existant ; le recrée s'il est absent, vide ou corrompu."""
    os.makedirs(index_dir, exist_ok=True)
    try:
        return open_dir(index_dir)
    except Exception as exc:  # index absent, vide ou corrompu -> recréation
        logger.warning("Index %s illisible (%s), recréation à neuf", index_dir, exc)
        try:
            shutil.rmtree(index_dir)
        except OSError:
            logger.warning("Impossible de supprimer le dossier d'index %s", index_dir)
        os.makedirs(index_dir, exist_ok=True)
        return create_in(index_dir, schema)


def open_main_index(index_dir: str) -> WhooshIndex:
    """Ouvre (ou recrée) l'index principal conversations/mémoire/fichiers."""
    return _open_or_recreate(index_dir, get_schema())


def open_skills_index(index_dir: str) -> WhooshIndex:
    """Ouvre (ou recrée) l'index séparé des skills Markdown."""
    return _open_or_recreate(index_dir, get_schema())


def _normalize_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(str(value))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def needs_reindex(index: WhooshIndex, doc_id: str, new_hash: str) -> bool:
    """Retourne True si le document ``doc_id`` est absent ou si son hash a changé."""
    with index.searcher() as searcher:
        doc = searcher.stored_fields(doc_id)
    if doc is None:
        return True
    return doc.get("content_hash") != new_hash


def index_document(index: WhooshIndex, *, id: str, type: str, source: str,
                   title: str = "", content: str = "", timestamp: Any = None,
                   session_id: str = "", role: str = "", path: str = "",
                   skill_name: str = "", metadata: Optional[dict] = None,
                   content_hash_value: Optional[str] = None,
                   embedding: Optional[Sequence[float]] = None) -> str:
    """Ajoute ou remplace (même ``id`` unique) un document dans l'index.

    Écriture via le writer whoosh-ng en context manager : commit automatique
    à la sortie du bloc ``with``, rollback implicite en cas d'exception.

    Args:
        embedding: vecteur de dimension ``dimension`` (champ ``embedding``,
            colonne VECTOR native). ``None`` => champ absent (BM25 seul).
    """
    if metadata is None:
        metadata = {}
    if content_hash_value is None:
        content_hash_value = content_hash(content)
    ts = _normalize_timestamp(timestamp) if timestamp is not None else datetime.now(timezone.utc)

    fields = {
        "id": str(id),
        "type": str(type),
        "source": str(source),
        "title": title or "",
        "content": content or "",
        "timestamp": ts,
        "session_id": str(session_id),
        "path": str(path),
        "skill_name": str(skill_name),
        "metadata": metadata,
        "content_hash": content_hash_value,
    }
    if role:
        fields["role"] = str(role)
    if embedding is not None:
        fields["embedding"] = list(embedding)

    with index.writer() as writer:
        writer.update_document(**fields)

    logger.info(
        "Document indexé id=%s type=%s source=%s hash=%s",
        id, type, source, content_hash_value[:12],
    )
    return content_hash_value


def index_turn(index: WhooshIndex, *, id: str, source: str, role: str,
               title: str, content: str, timestamp: Any, session_id: str,
               content_hash_value: Optional[str] = None,
               type: str = TYPE_SESSION_TURN, path: str = "",
               skill_name: str = "", metadata: Optional[dict] = None) -> str:
    """Indexe (ou remplace) un turn de conversation (type ``session_turn``).

    Champs obligatoires : id, source, role, title, content, timestamp, session_id.
    ``content_hash`` est calculé (SHA-256 du contenu) s'il n'est pas fourni.
    """
    return index_document(
        index,
        id=str(id),
        type=type,
        source=str(source),
        role=str(role),
        title=title,
        content=content,
        timestamp=timestamp,
        session_id=str(session_id),
        path=path,
        skill_name=skill_name,
        metadata=metadata,
        content_hash_value=content_hash_value,
    )