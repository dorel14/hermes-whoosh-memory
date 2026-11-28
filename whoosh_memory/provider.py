"""Fournisseur de mémoire ``whoosh-memory`` pour Hermes Agent.

Implémente ``agent.memory_provider.MemoryProvider`` au-dessus d'un index
whoosh-ng local (repli BM25 pur Python si whoosh-ng est absent).

Contrat respecté (voir docs/HERMES_CONTRACT.md) :

- tous les chemins dérivent du ``hermes_home`` reçu par ``initialize()`` ;
  rien n'est gardé au niveau du module ou de la classe (isolation des profils) ;
- ``sync_turn()`` ne bloque pas : les écritures passent par une file traitée
  par un unique thread créé avec ``spawn_context_thread`` ;
- aucune écriture automatique quand ``agent_context`` n'est pas ``primary``
  (cron, sous-agent, flush) ;
- ``handle_tool_call()`` renvoie toujours une chaîne JSON, erreurs comprises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from agent.memory_provider import (
    MemoryProvider,
    RecallStatus,
    is_trivial_prompt,
    spawn_context_thread,
)

from .config import load_config
from .search import whoosh_search
from .tools import whoosh_add, whoosh_forget, whoosh_open, whoosh_stats

logger = logging.getLogger(__name__)

__all__ = ["WhooshMemoryProvider", "register", "PROVIDER_NAME"]

PROVIDER_NAME = "whoosh-memory"

#: Délai maximal d'attente du thread d'écriture (flush / arrêt), en secondes.
_FLUSH_TIMEOUT = 10.0


def register(ctx) -> None:
    """Point d'entrée appelé par Hermes : enregistre une NOUVELLE instance."""
    ctx.register_memory_provider(WhooshMemoryProvider())


def _memory_doc_id(target: str, content: str) -> str:
    """Identifiant stable d'une entrée MEMORY.md / USER.md (dérivé du contenu)."""
    digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:20]
    return f"memory:{target}:{digest}"


class WhooshMemoryProvider(MemoryProvider):
    """Mémoire locale hybride (BM25 + vecteurs optionnels) sur whoosh-ng."""

    def __init__(self) -> None:
        self._hermes_home: Optional[Path] = None
        self._data_dir: Optional[Path] = None
        self._index_dir: Optional[Path] = None
        self._config: dict[str, Any] = {}
        self._session_id = ""
        self._writes_enabled = False
        self._initialized = False
        self._last_recall_count: Optional[int] = None
        # Un seul verrou pour toutes les opérations d'index (lecture et écriture) :
        # whoosh n'autorise qu'un writer à la fois.
        self._index_lock = threading.Lock()
        self._queue: "queue.Queue[Optional[Callable[[], None]]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

    # -- identité / disponibilité --------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """Toujours disponible : whoosh-ng si présent, sinon BM25 pur. Aucun réseau."""
        return True

    # -- cycle de vie ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            raise ValueError("whoosh-memory: initialize() requiert hermes_home")
        self._hermes_home = Path(hermes_home)
        self._data_dir = self._hermes_home / PROVIDER_NAME
        self._index_dir = self._data_dir / "index"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._config = load_config(self._data_dir / "config.json")
        self._session_id = session_id or ""
        agent_context = kwargs.get("agent_context") or "primary"
        self._writes_enabled = agent_context == "primary"
        if self._writes_enabled:
            self._worker = spawn_context_thread(self._drain, name="whoosh-memory-writer")
            self._worker.start()
        self._initialized = True
        logger.info(
            "whoosh-memory: initialisé (context=%s, index=%s)", agent_context, self._index_dir
        )

    def shutdown(self) -> None:
        if self._worker is not None:
            self._queue.put(None)  # sentinelle d'arrêt
            self._worker.join(timeout=_FLUSH_TIMEOUT)
            self._worker = None
        self._initialized = False

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        self._flush()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        if reset:
            self._flush()
        self._session_id = new_session_id or ""

    # -- file d'écriture (thread unique) ---------------------------------------

    def _drain(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                with self._index_lock:
                    job()
            except Exception as exc:  # une écriture ratée ne doit jamais tuer le thread
                logger.warning("whoosh-memory: écriture échouée: %s", exc)
            finally:
                self._queue.task_done()

    def _submit(self, job: Callable[[], None]) -> None:
        if self._writes_enabled and self._worker is not None:
            self._queue.put(job)

    def _flush(self) -> None:
        """Attend (avec délai max) que les écritures en file soient appliquées."""
        if self._worker is None:
            return
        done = threading.Event()
        self._queue.put(done.set)
        done.wait(timeout=_FLUSH_TIMEOUT)

    # -- prompt / rappel --------------------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "Une mémoire locale whoosh-memory indexe les conversations passées et les "
            "entrées de mémoire. Utilise whoosh_search pour y retrouver une information "
            "ancienne, puis whoosh_open pour lire un résultat en entier."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall_count = None
        if not self._initialized or is_trivial_prompt(query):
            return ""
        top_k = int(self._config.get("prefetch_top_k", 3))
        max_chars = int(self._config.get("prefetch_max_chars", 3000))
        try:
            with self._index_lock:
                hits = whoosh_search(str(self._index_dir), query, top_k=top_k)
        except Exception as exc:  # le rappel ne doit jamais bloquer un tour
            logger.debug("whoosh-memory: prefetch échoué: %s", exc)
            return ""
        if not hits:
            return ""
        lines = ["Contexte retrouvé dans la mémoire locale (whoosh-memory) :"]
        used = len(lines[0])
        count = 0
        for hit in hits:
            line = f"- [{hit.doc_id}] {hit.title}: {(hit.snippet or '').strip()}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
            count += 1
        if not count:
            return ""
        self._last_recall_count = count
        return "\n".join(lines)

    def recall_status(self) -> Optional[RecallStatus]:
        if not self._last_recall_count:
            return None
        return RecallStatus(provider_label=PROVIDER_NAME, count=self._last_recall_count)

    # -- écritures automatiques -------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[list[dict[str, Any]]] = None,
        turn_author: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self._config.get("index_sessions", True):
            return
        text = f"Utilisateur : {user_content or ''}\nAssistant : {assistant_content or ''}"
        if not (user_content or assistant_content):
            return
        sid = session_id or self._session_id
        doc_id = f"turn:{sid}:{uuid.uuid4().hex[:12]}"
        index_dir = str(self._index_dir)
        self._submit(
            lambda: whoosh_add(
                index_dir,
                id=doc_id,
                path=f"session:{sid}",
                title=f"Conversation {sid}",
                content=text,
                type="session_turn",
            )
        )

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Reflète une écriture de la mémoire intégrée (add | replace | remove)."""
        previous = (metadata or {}).get("previous_content") or ""
        index_dir = str(self._index_dir)
        path = f"memory:{target}"
        title = "Profil utilisateur" if target == "user" else "Mémoire"

        def job() -> None:
            if action in ("replace", "remove"):
                old = previous or (content if action == "remove" else "")
                if old:
                    whoosh_forget(index_dir, id=_memory_doc_id(target, old))
            if action in ("add", "replace") and content:
                whoosh_add(
                    index_dir,
                    id=_memory_doc_id(target, content),
                    path=path,
                    title=title,
                    content=content,
                    type="memory",
                )

        self._submit(job)

    # -- outils -------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "whoosh_search",
                "description": "Recherche plein texte (BM25) dans la mémoire locale whoosh-memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Texte recherché."},
                        "limit": {
                            "type": "integer",
                            "description": "Nombre de résultats (1 à 20, défaut 5).",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_open",
                "description": "Lit un document de la mémoire par id (ou path), page par page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "id renvoyé par whoosh_search."},
                        "path": {"type": "string", "description": "path du document."},
                        "offset": {"type": "integer", "minimum": 0, "description": "Défaut 0."},
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20000,
                            "description": "Défaut 4000.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_add",
                "description": "Enregistre explicitement une note dans la mémoire locale.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Titre court."},
                        "content": {"type": "string", "description": "Texte à mémoriser."},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_forget",
                "description": "Supprime un document par id (ou tous ceux d'un path).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "whoosh_stats",
                "description": "Statistiques de la mémoire locale (nombre de documents par type).",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        try:
            result = self._dispatch(tool_name, args or {})
        except Exception as exc:
            logger.warning("whoosh-memory: outil %s en échec: %s", tool_name, exc)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False, default=str)

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> Any:
        if not self._initialized:
            return {"error": "whoosh-memory n'est pas initialisé"}
        index_dir = str(self._index_dir)

        if tool_name == "whoosh_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query est obligatoire"}
            limit = min(max(int(args.get("limit", 5)), 1), 20)
            with self._index_lock:
                hits = whoosh_search(index_dir, query, top_k=limit)
            return {"results": [asdict(h) for h in hits]}

        if tool_name == "whoosh_open":
            if not (args.get("id") or args.get("path")):
                return {"error": "id ou path est obligatoire"}
            with self._index_lock:
                doc = whoosh_open(
                    index_dir,
                    id=args.get("id"),
                    path=args.get("path"),
                    offset=int(args.get("offset", 0)),
                    max_chars=min(int(args.get("max_chars", 4000)), 20000),
                )
            return doc if doc is not None else {"error": "document introuvable"}

        if tool_name == "whoosh_add":
            content = str(args.get("content") or "").strip()
            if not content:
                return {"error": "content est obligatoire"}
            doc_id = f"explicit:{uuid.uuid4().hex[:16]}"
            self._flush()
            with self._index_lock:
                whoosh_add(
                    index_dir,
                    id=doc_id,
                    path=doc_id,
                    title=str(args.get("title") or "Note"),
                    content=content,
                    type="explicit",
                )
            return {"ok": True, "id": doc_id}

        if tool_name == "whoosh_forget":
            if not (args.get("id") or args.get("path")):
                return {"error": "id ou path est obligatoire"}
            self._flush()
            with self._index_lock:
                removed = whoosh_forget(index_dir, id=args.get("id"), path=args.get("path"))
            return {"removed": removed}

        if tool_name == "whoosh_stats":
            with self._index_lock:
                stats = whoosh_stats(index_dir)
            return stats

        return {"error": f"outil inconnu : {tool_name}"}

    # -- configuration -------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "index_sessions",
                "description": "Indexer chaque tour de conversation.",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "prefetch_top_k",
                "description": "Nombre de souvenirs injectés avant chaque tour.",
                "type": "integer",
                "default": 3,
                "minimum": 0,
                "maximum": 10,
            },
            {
                "key": "prefetch_max_chars",
                "description": "Budget de caractères du contexte injecté.",
                "type": "integer",
                "default": 3000,
                "minimum": 200,
                "maximum": 20000,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / PROVIDER_NAME / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                current = {}
        except (OSError, ValueError):
            current = {}
        current.update(values or {})
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        if self._hermes_home is not None and Path(hermes_home) == self._hermes_home:
            self._config.update(values or {})
