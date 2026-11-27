"""Whoosh-memory memory provider for Hermes Agent.

Implements the ``MemoryProvider`` interface (agent/memory_provider.py) using a
local whoosh-ng index (BM25 + optional vector). All paths derive from
``hermes_home`` received in ``initialize()`` — never hardcoded.

Optional dependencies (embeddings, images) are imported lazily inside the
functions that need them, per R5 of the contract.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider, spawn_context_thread, RecallStatus

from .search import SearchHit, _Index, whoosh_search, index_documents
from .tools import whoosh_open, whoosh_forget, whoosh_stats, whoosh_add
from .config import DEFAULTS, load_config

logger = logging.getLogger(__name__)

__all__ = ["WhooshMemoryProvider", "register"]


def register(ctx) -> None:
    """Plugin entry point called by Hermes at load time.

    Registers a fresh :class:`WhooshMemoryProvider` instance via
    ``ctx.register_memory_provider``. A new instance is created on every call
    so concurrent sessions never share state.
    """
    provider = WhooshMemoryProvider()
    ctx.register_memory_provider(provider)
    logger.info("whoosh-memory: MemoryProvider enregistré")


class WhooshMemoryProvider(MemoryProvider):
    """Local hybrid (BM25 + vector) memory provider backed by whoosh-ng."""

    pre_compress_checkpoint_api_version = 1

    def __init__(self) -> None:
        self._hermes_home: Optional[Path] = None
        self._index_dir: Optional[Path] = None
        self._index: Optional[_Index] = None
        self._session_id: Optional[str] = None
        self._config: dict[str, Any] = {}
        self._initialized = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "whoosh-memory"

    def is_available(self) -> bool:
        """Always available: whoosh-ng if present, else pure-Python BM25."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if hermes_home is None:
            raise TypeError("whoosh-memory: initialize() requires hermes_home")
        self._hermes_home = Path(hermes_home)
        self._index_dir = self._hermes_home / "plugin-data" / "whoosh-memory" / "index"
        self._session_id = session_id
        self._config = dict(DEFAULTS)
        cfg_path = self._hermes_home / "plugin-data" / "whoosh-memory" / "config.json"
        try:
            self._config = load_config(cfg_path)
        except Exception:  # pragma: no cover - config never crashes the provider
            logger.debug("whoosh-memory: config non chargé, defaults utilisés")
        # Open the index lazily on first use; _Index handles missing dir.
        self._index = _Index(self._index_dir)
        self._index.finish()
        self._initialized = True
        logger.info("whoosh-memory: initialisé (session=%s, index=%s)", session_id, self._index_dir)

    def unavailable_reason(self) -> str:
        return ""

    def system_prompt_block(self) -> str:
        return (
            "Vous disposez d'un index mémoire local whoosh-memory (BM25 + "
            "optionnel vectoriel). Utilisez whoosh_search pour retrouver des "
            "documents passés, whoosh_open pour les lire, whoosh_forget pour "
            "les supprimer, whoosh_stats pour les statistiques et whoosh_add "
            "pour ajouter des documents."
        )

    # -- recall / prefetch ---------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return formatted recall context for the upcoming turn.

        Runs a whoosh_search and returns a short, human-readable block.
        Must be fast: results are cached from the open index.
        """
        if not query or not self._initialized:
            return ""
        try:
            top_k = int(self._config.get("prefetch_top_k", 3))
            hits = whoosh_search(str(self._index_dir), query, top_k=top_k)
        except Exception as exc:  # pragma: no cover - recall never blocks a turn
            logger.debug("whoosh-memory: prefetch échec: %s", exc)
            return ""
        if not hits:
            return ""
        lines = ["### Contexte mémoire (whoosh-memory)"]
        for h in hits:
            snippet = (h.snippet or h.title or "")[:200]
            lines.append(f"- {h.title or h.path or h.doc_id}: {snippet}")
        return "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch is synchronous and fast enough to not queue."""
        return None

    def recall_status(self) -> Optional[RecallStatus]:
        if not self._initialized or self._index is None:
            return None
        try:
            stats = whoosh_stats(str(self._index_dir))
        except Exception:
            return None
        count = stats.get("documents", 0)
        return RecallStatus(
            provider_label=self.name,
            count=count,
        )

    # -- turn sync -----------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[list[dict[str, Any]]] = None,
        turn_author: Optional[dict[str, Any]] = None,
    ) -> None:
        """Index the current conversation turn into the whoosh index."""
        if not self._initialized:
            return
        text = f"Utilisateur: {user_content}\nAssistant: {assistant_content}".strip()
        if not text:
            return
        doc_id = f"turn:{session_id or self._session_id}:{os.urandom(4).hex()}"
        try:
            whoosh_add(
                str(self._index_dir),
                id=doc_id,
                path=f"session:{session_id or self._session_id}",
                title="Tour de conversation",
                content=text,
                type="conversation",
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("whoosh-memory: sync_turn échec: %s", exc)

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for the whoosh-memory tools."""
        return [
            {
                "name": "whoosh_search",
                "description": "Recherche dans l'index mémoire whoosh-memory (BM25, optionnel hybride/vectoriel).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Texte de la recherche."},
                        "top_k": {"type": "integer", "description": "Nombre de résultats (défaut 10).", "default": 10},
                        "hybrid": {
                            "type": "string",
                            "description": "'auto' | 'true' | 'false' — mode de recherche (défaut 'auto').",
                            "default": "auto",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_open",
                "description": "Lit un document indexé par id ou path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Identifiant du document."},
                        "path": {"type": "string", "description": "Chemin du document."},
                        "offset": {"type": "integer", "description": "Décalage en caractères (défaut 0).", "default": 0},
                        "max_chars": {
                            "type": "integer",
                            "description": "Nombre max de caractères à renvoyer (défaut 4000).",
                            "default": 4000,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_forget",
                "description": "Supprime un document de l'index par id ou path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Identifiant du document à supprimer."},
                        "path": {"type": "string", "description": "Chemin du document à supprimer."},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_stats",
                "description": "Renvoie les statistiques de l'index whoosh-memory.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_add",
                "description": "Ajoute un document manuellement à l'index.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Identifiant du document."},
                        "path": {"type": "string", "description": "Chemin du document."},
                        "title": {"type": "string", "description": "Titre du document."},
                        "content": {"type": "string", "description": "Contenu texte à indexer."},
                        "type": {
                            "type": "string",
                            "description": "Type de document (défaut 'explicit').",
                            "default": "explicit",
                        },
                    },
                    "required": ["id", "path", "title", "content"],
                    "additionalProperties": False,
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        """Handle one whoosh-memory tool; returns a JSON string."""
        index_dir = str(self._index_dir) if self._index_dir else ""

        if tool_name == "whoosh_search":
            query = args.get("query", "")
            top_k = int(args.get("top_k", 10))
            hybrid = args.get("hybrid", "auto")
            hits = whoosh_search(index_dir, query, top_k=top_k, hybrid=hybrid)
            return json.dumps([asdict(h) for h in hits], ensure_ascii=False)

        if tool_name == "whoosh_open":
            result = whoosh_open(
                index_dir,
                id=args.get("id"),
                path=args.get("path"),
                offset=int(args.get("offset", 0)),
                max_chars=int(args.get("max_chars", 4000)),
            )
            return json.dumps(result, ensure_ascii=False)

        if tool_name == "whoosh_forget":
            result = whoosh_forget(
                index_dir,
                id=args.get("id"),
                path=args.get("path"),
            )
            return json.dumps(result, ensure_ascii=False)

        if tool_name == "whoosh_stats":
            result = whoosh_stats(index_dir)
            return json.dumps(result, ensure_ascii=False)

        if tool_name == "whoosh_add":
            result = whoosh_add(
                index_dir,
                id=args["id"],
                path=args["path"],
                title=args.get("title", ""),
                content=args.get("content", ""),
                type=args.get("type", "explicit"),
            )
            return json.dumps(result, ensure_ascii=False)

        raise NotImplementedError(
            f"whoosh-memory: outil inconnu '{tool_name}'"
        )

    # -- lifecycle hooks -----------------------------------------------------

    def shutdown(self) -> None:
        self._index = None
        self._initialized = False

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        return None

    def identity_signature(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "index_dir": str(self._index_dir) if self._index_dir else "",
            "initialized": self._initialized,
        }

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Persist the index at end of session."""
        if self._index is not None:
            self._index.finish()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id
        if reset and self._index is not None:
            self._index.finish()

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Checkpoint: flush the index before context compression."""
        if self._index is not None:
            self._index.finish()
        return ""

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs
    ) -> None:
        return None

    # -- config --------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "index_sessions",
                "description": "Indexer l'historique des sessions de conversation.",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "index_tool_results",
                "description": "Indexer les résultats d'outils agents.",
                "type": "boolean",
                "default": False,
            },
            {
                "key": "index_paths",
                "description": "Chemins de fichiers locaux à indexer.",
                "type": "text",
                "default": "~/Documents, ~/notes",
            },
            {
                "key": "prefetch_top_k",
                "description": "Nombre de résultats retournés par prefetch.",
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 20,
            },
            {
                "key": "chunk_size",
                "description": "Taille des chunks d'indexation (caractères).",
                "type": "integer",
                "default": 1000,
                "minimum": 100,
                "maximum": 5000,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        cfg_path = Path(hermes_home) / "plugin-data" / "whoosh-memory" / "config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
        self._config = dict(values)

    def read_config_fields(self, hermes_home: str) -> dict[str, Any]:
        cfg_path = Path(hermes_home) / "plugin-data" / "whoosh-memory" / "config.json"
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return {"values": data, "placeholders": {}}

    def write_config_fields(self, values: dict[str, Any], hermes_home: str) -> None:
        self.save_config(values, hermes_home)

    # -- memory write mirror -------------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Mirror MEMORY.md / USER.md writes into the whoosh index."""
        if not self._initialized or not content:
            return
        doc_id = f"memory:{target}:{os.urandom(4).hex()}"
        try:
            whoosh_add(
                str(self._index_dir),
                id=doc_id,
                path=f"memory:{target}",
                title=f"Mémoire {target}",
                content=content,
                type="memory",
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("whoosh-memory: on_memory_write échec: %s", exc)

    # -- backup --------------------------------------------------------------

    def backup_paths(self) -> list[str]:
        if self._index_dir is None:
            return []
        return [str(self._index_dir)]