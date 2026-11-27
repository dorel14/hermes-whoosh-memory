"""Embeddings locaux optionnels + fusion hybride pour whoosh-memory.

Section 7.2 — Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~
Providers : ``fastembed`` (CPU/ONNX, zéro PyTorch, recommandé),
``sentence-transformers`` (PyTorch) et ``Ollama`` (HTTP local).
Modèles multilingues français pris en charge par alias : ``bge-m3``,
``multilingual-e5`` et ``multilingual-e5-small`` (dimension configurée,
défaut 384).

Le cache disque est clé par hache de contenu (SHA-256 de
``provider|modèle|dim|texte``) : un texte inchangé n'est jamais ré-encodé
entre deux exécutions.

Si aucun provider n'est disponible (ou ``embedding_enabled = false``), le
plugin fonctionne en mode dégradé : recherche BM25 pure, sans vecteur,
sans erreur et sans appel réseau.

Section 7.3 — Fusion hybride
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``HybridFuser`` combine une liste de résultats BM25 et une liste de
résultats vectoriels selon deux stratégies :

- RRF (reciprocal rank fusion) : ``score += 1 / (k + rang)`` avec
  ``k = 60`` par défaut.
- Fusion linéaire pondérée :
  ``alpha * score_bm25_normalise + (1 - alpha) * score_vector_normalise``
  avec normalisation min-max de chaque liste.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import hashlib
import json
import logging
import math
import os
import threading
import urllib.error
import urllib.request

logger = logging.getLogger("hermes.plugins.whoosh-memory.embeddings")

DEFAULT_DIM = 384
DEFAULT_RRF_K = 60

# Alias modèles multilingues FR -> (identifiant provider, dimension native).
FR_MODELS: dict[str, tuple[str, int]] = {
    "bge-m3": ("BAAI/bge-m3", 1024),
    "multilingual-e5": ("intfloat/multilingual-e5-base", 768),
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", 384),
}

# No global hardcoded HERMES_HOME; resolve via provider.initialize().
# DEFAULT_CACHE_DIR is constructed by EmbeddingConfig via provider._hermes_home.


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration des embeddings (clés de config.json de whoosh-memory)."""

    provider: str = "auto"            # auto | fastembed | sentence-transformers | ollama
    model: str = "multilingual-e5-small"
    dim: int = DEFAULT_DIM
    cache_dir: Path | None = None
    cache_enabled: bool = True
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 10.0
    normalize: bool = True
    enabled: bool = True              # False -> mode degrade BM25 pur dès l'initialisation

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(os.path.expanduser(str(self.cache_dir))))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None, hermes_home: Path) -> "EmbeddingConfig":
        """Construit la config depuis un dict de type ``whoosh_memory.config``."""

        def get(key: str, default: Any) -> Any:
            return cfg.get(key, default) if isinstance(cfg, Mapping) else default

        dim = get("embedding_dim", DEFAULT_DIM)
        try:
            dim = int(dim)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            dim = DEFAULT_DIM
        
        default_cache = hermes_home / "plugin-data" / "whoosh-memory" / "embeddings_cache"
        cache_dir = get("embedding_cache_dir", default_cache)
        
        return cls(
            provider=str(get("embedding_provider", "auto")),
            model=str(get("embedding_model", "multilingual-e5-small")),
            dim=dim,
            cache_dir=Path(os.path.expanduser(str(cache_dir))),
            cache_enabled=bool(get("embedding_cache", True)),
            ollama_host=str(get("ollama_host", "http://localhost:11434")),
            ollama_timeout=float(get("ollama_timeout", 10.0)),
            normalize=bool(get("embedding_normalize", True)),
            enabled=bool(get("embedding_enabled", True)),
        )


def _resolve_model(model: str) -> str:
    """Résout un alias FR (bge-m3, multilingual-e5, ...) vers l'identifiant du provider."""
    return FR_MODELS.get(model, (model,))[0]


def content_hash(text: str, provider: str, model: str, dim: int) -> str:
    """SHA-256 de ``provider|modèle|dim|texte`` — clef du cache d'embeddings."""
    payload = f"{provider}|{model}|{dim}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_dim(vector: Sequence[float], dim: int) -> list[float]:
    """Tronque ou complète par des zéros pour obtenir exactement ``dim`` valeurs."""
    vals = [float(x) for x in vector]
    if len(vals) >= dim:
        return vals[:dim]
    return vals + [0.0] * (dim - len(vals))


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Normalisation L2 ; rend le vecteur inchangé si sa norme est nulle."""
    vals = [float(x) for x in vector]
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 0.0:
        return list(vals)
    return [v / norm for v in vals]


# --------------------------------------------------------------------------- #
# Providers d'embeddings
# --------------------------------------------------------------------------- #

class EmbeddingProvider(Protocol):
    """Un provider expose sa dimension etencode un texte en vecteur de floats."""

    @property
    def dimension(self) -> int:
        ...

    @property
    def native_dimension(self) -> int | None:
        ...

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class FastEmbedProvider:
    """Provider CPU/ONNX basé sur ``fastembed.TextEmbedding``."""

    def __init__(
        self,
        model: str,
        dim: int | None = None,
        threads: int | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed est requis pour le provider 'fastembed'. "
                "Installez-le : pip install fastembed"
            ) from exc
        resolved = _resolve_model(model)
        kwargs: dict[str, Any] = {}
        if threads is not None:
            kwargs["threads"] = threads
        self._model = TextEmbedding(resolved, **kwargs)
        native = int(getattr(self._model, "dim", 0) or 0)
        self._native_dim = native or self._probe_dim()
        self._dim = dim or self._native_dim

    def _probe_dim(self) -> int:
        first = next(iter(self._model.embed(["__whoosh_dim_probe__"])))
        return len([float(x) for x in first.tolist()])

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def native_dimension(self) -> int | None:
        return self._native_dim

    def embed(self, text: str) -> list[float]:
        result = self._model.embed([text])
        first = result[0] if isinstance(result, list) else next(iter(result))
        return [float(x) for x in first.tolist()]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for vec in self._model.embed(list(texts)):
            out.append([float(x) for x in vec.tolist()])
        return out


class SentenceTransformersProvider:
    """Provider PyTorch basé sur ``sentence-transformers``."""

    def __init__(
        self,
        model: str,
        dim: int | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            import sentence_transformers  # noqa: F401  (presence check)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers est requis pour le provider "
                "'sentence-transformers'. Installez-le : "
                "pip install sentence-transformers"
            ) from exc
        from sentence_transformers import SentenceTransformer

        resolved = _resolve_model(model)
        self._model = SentenceTransformer(resolved)
        self._native_dim = int(self._model.get_sentence_embedding_dimension())
        self._dim = dim or self._native_dim
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def native_dimension(self) -> int | None:
        return self._native_dim

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=self._normalize
        )
        return [float(v) for v in vec.tolist()]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=self._normalize
        )
        return [row.tolist() for row in arr]


class OllamaProvider:
    """Provider HTTP appelant ``POST {host}/api/embeddings`` d'Ollama."""

    def __init__(
        self,
        model: str,
        host: str,
        timeout: float,
        dim: int | None = None,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._dim = dim or DEFAULT_DIM
        self._native_dim: int | None = dim

    def _call(self, text: str) -> list[float]:
        payload = json.dumps({"model": self._model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            self._host + "/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vec = data.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError(
                f"Réponse Ollama invalide pour le modèle {self._model!r}"
            )
        if self._native_dim is None:
            self._native_dim = len(vec)
        return [float(x) for x in vec]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def native_dimension(self) -> int | None:
        return self._native_dim

    def embed(self, text: str) -> list[float]:
        return self._call(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._call(t) for t in texts]


# --------------------------------------------------------------------------- #
# Point d'entree : encodage + cache + degradé BM25 pur
# --------------------------------------------------------------------------- #

class Embeddings:
    """Encodage de textes avec cache disque et degradation gracieuse.

    Attributes:
        available: True si un provider est opérationnel.
        provider_name: nom du provider actif (``fastembed``, ...).
        dim: dimension de sortie configurée.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config if config is not None else EmbeddingConfig(**kwargs)
        self._provider: EmbeddingProvider | None = None
        self._provider_name: str | None = None
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.RLock()
        self._cache_path = self.config.cache_dir / "cache.json"
        self._load_cache()
        if self.config.enabled:
            self._provider, self._provider_name = self._build_provider()
            if self._provider is None:
                logger.info(
                    "Aucun provider d'embeddings disponible — mode degrade BM25 pur."
                )
        else:
            logger.info(
                "Embeddings desactives par la configuration — mode degrade BM25 pur."
            )

    # -- constructeurs ------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Embeddings":
        """Construit depuis un dict de config whoosh-memory (spec section 8)."""
        return cls(EmbeddingConfig.from_config(cfg))

    @classmethod
    def from_env(cls) -> "Embeddings":
        """Construit depuis les variables d'environnement ``WHOOSH_MEMORY_*``."""
        kwargs: dict[str, Any] = {}
        env_map = {
            "WHOOSH_MEMORY_PROVIDER": "provider",
            "WHOOSH_MEMORY_MODEL": "model",
            "WHOOSH_MEMORY_DIM": "dim",
            "WHOOSH_MEMORY_CACHE_DIR": "cache_dir",
            "WHOOSH_MEMORY_CACHE": "cache_enabled",
            "WHOOSH_MEMORY_OLLAMA_HOST": "ollama_host",
            "WHOOSH_MEMORY_TIMEOUT": "ollama_timeout",
            "WHOOSH_MEMORY_NORMALIZE": "normalize",
            "WHOOSH_MEMORY_ENABLED": "enabled",
        }
        for env, attr in env_map.items():
            val = os.environ.get(env)
            if val is None:
                continue
            if attr == "dim":
                try:
                    kwargs[attr] = int(val)
                except ValueError:
                    logger.debug("Valeur invalide pour %s: %r", env, val)
            elif attr in ("cache_enabled", "normalize", "enabled"):
                kwargs[attr] = val.lower() not in ("0", "false", "no", "")
            elif attr == "ollama_timeout":
                try:
                    kwargs[attr] = float(val)
                except ValueError:
                    logger.debug("Valeur invalide pour %s: %r", env, val)
            else:
                kwargs[attr] = val
        return cls(EmbeddingConfig(**kwargs))

    # -- providers ----------------------------------------------------------

    def _build_provider(self) -> tuple[EmbeddingProvider | None, str | None]:
        name = str(self.config.provider).lower()
        if name == "auto":
            candidates = [
                ("fastembed", self._make_fastembed),
                ("sentence-transformers", self._make_sentence_transformers),
                ("ollama", self._make_ollama),
            ]
        elif name == "fastembed":
            candidates = [("fastembed", self._make_fastembed)]
        elif name in ("sentence-transformers", "sentence_transformers", "st"):
            candidates = [("sentence-transformers", self._make_sentence_transformers)]
        elif name == "ollama":
            candidates = [("ollama", self._make_ollama)]
        else:
            raise ValueError(
                f"Provider d'embeddings inconnu: {self.config.provider!r}"
            )
        for label, factory in candidates:
            try:
                provider = factory()
            except ImportError as exc:
                logger.debug("Provider %s indisponible: %s", label, exc)
                continue
            except Exception as exc:  # Ollama non joignable, etc.
                logger.warning("Provider %s ignore (%s)", label, exc)
                continue
            if provider is not None:
                native = provider.native_dimension or provider.dimension
                logger.info(
                    "Embeddings : provider %s, modele %s (native %d dims, "
                    "sortie configuree %d dims)",
                    label,
                    self.config.model,
                    native,
                    self.config.dim,
                )
                return provider, label
        return None, None

    def _make_fastembed(self) -> FastEmbedProvider:
        return FastEmbedProvider(self.config.model, dim=self.config.dim)

    def _make_sentence_transformers(self) -> SentenceTransformersProvider:
        return SentenceTransformersProvider(
            self.config.model, dim=self.config.dim, normalize=self.config.normalize
        )

    def _make_ollama(self) -> OllamaProvider:
        return OllamaProvider(
            self.config.model,
            self.config.ollama_host,
            self.config.ollama_timeout,
            dim=self.config.dim,
        )

    # -- API d'encodage -----------------------------------------------------

    @property
    def available(self) -> bool:
        return self._provider is not None

    @property
    def provider_name(self) -> str | None:
        return self._provider_name

    @property
    def dim(self) -> int:
        return self.config.dim

    def embed(self, text: str) -> list[float] | None:
        """Encode ``text`` en vecteur de ``config.dim`` floats.

        Retourne ``None`` en mode degrade (provider indisponible ou echec a
        l'execution). Un texte vide renvoie un vecteur nul de la bonne
        dimension.
        """
        if not text:
            return [0.0] * self.config.dim
        if not self.available:
            return None
        key = content_hash(
            text, self._provider_name, self.config.model, self.config.dim
        )
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return list(cached)
        assert self._provider is not None
        try:
            vec = self._provider.embed(text)
        except Exception as exc:  # provider defaillant -> degrade sur ce texte
            logger.warning(
                "Echec d'encodage (texte ignore pour les vecteurs): %s", exc
            )
            return None
        vec = to_dim(vec, self.config.dim)
        if self.config.normalize:
            vec = l2_normalize(vec)
        if self.config.cache_enabled:
            with self._lock:
                self._cache[key] = list(vec)
                self._persist_cache()
        return list(vec)

    def embed_batch(
        self, texts: Sequence[str]
    ) -> list[list[float] | None]:
        """Encode un lot de textes (cache respecté, degradé gracieux)."""
        results: list[list[float] | None] = [None] * len(texts)
        todo: list[tuple[int, str, str]] = []  # (index, texte, cle cache)
        for i, text in enumerate(texts):
            if not text:
                results[i] = [0.0] * self.config.dim
                continue
            if not self.available:
                results[i] = None
                continue
            key = content_hash(
                text, self._provider_name, self.config.model, self.config.dim
            )
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                results[i] = list(cached)
            else:
                todo.append((i, text, key))
        if todo and self.available:
            assert self._provider is not None
            try:
                vecs = self._provider.embed_batch([t for _, t, _ in todo])
            except Exception as exc:
                logger.warning("Echec d'encodage par lot: %s", exc)
                vecs = [None] * len(todo)
            for (i, _text, key), vec in zip(todo, vecs):
                if vec is None:
                    continue
                vec = to_dim(vec, self.config.dim)
                if self.config.normalize:
                    vec = l2_normalize(vec)
                results[i] = list(vec)
                if self.config.cache_enabled:
                    with self._lock:
                        self._cache[key] = list(vec)
        if self.config.cache_enabled:
            self._persist_cache()
        return results

    # -- cache disque -------------------------------------------------------

    def _load_cache(self) -> None:
        if not self.config.cache_enabled:
            return
        try:
            with self._cache_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(
                "Cache d'embeddings illisible (%s), reconstruction a neuf", exc
            )
            return
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    self._cache[k] = [float(x) for x in v]

    def _persist_cache(self) -> None:
        if not self.config.cache_enabled:
            return
        parent = self._cache_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_name(self._cache_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, ensure_ascii=False)
        tmp.replace(self._cache_path)

    def clear_cache(self) -> None:
        """Vide le cache en memoire et supprime le fichier disque."""
        with self._lock:
            self._cache.clear()
        try:
            self._cache_path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Fusion hybride (Section 7.3)
# --------------------------------------------------------------------------- #

@dataclass
class ScoredDoc:
    """Document score par un retrieveur (BM25 ou vectoriel)."""

    doc_id: str
    score: float
    rank: int = 0
    source: str = ""  # bm25 | vector | bm25+vector


class HybridFuser:
    """Fusion de resultats BM25 et vectoriels (RRF ou fusion ponderee)."""

    def __init__(
        self,
        mode: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if mode not in ("rrf", "weighted", "bm25", "vector"):
            raise ValueError(
                f"Mode de fusion inconnu: {mode!r} "
                "(attendu parmi rrf, weighted, bm25, vector)"
            )
        self.mode = mode
        self.alpha = alpha
        self.rrf_k = rrf_k

    def fuse(
        self,
        bm25: Sequence[ScoredDoc],
        vector: Sequence[ScoredDoc],
    ) -> list[ScoredDoc]:
        """Fusionne les deux listes et renvoie les documents scores tries."""
        bm25 = list(bm25)
        vector = list(vector)
        if self.mode == "bm25":
            return self._ranked(bm25)
        if self.mode == "vector":
            return self._ranked(vector)
        if self.mode == "rrf":
            return _rrf([bm25, vector], k=self.rrf_k)
        return _weighted(bm25, vector, alpha=self.alpha)

    @staticmethod
    def _ranked(docs: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        ordered = sorted(docs, key=lambda d: d.score, reverse=True)
        return [
            ScoredDoc(d.doc_id, d.score, i + 1, d.source) for i, d in enumerate(ordered)
        ]


def _rrf(
    lists: Sequence[Sequence[ScoredDoc]], k: int = DEFAULT_RRF_K
) -> list[ScoredDoc]:
    """Reciprocal Rank Fusion : score += 1 / (k + rang)."""
    scores: dict[str, float] = {}
    source_by_id: dict[str, str] = {}
    for lst in lists:
        for rank, doc in enumerate(lst):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank + 1)
            source_by_id.setdefault(doc.doc_id, doc.source)
    merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        ScoredDoc(doc_id=did, score=score, rank=i + 1, source=source_by_id.get(did, ""))
        for i, (did, score) in enumerate(merged)
    ]


def _weighted(
    bm25: Sequence[ScoredDoc],
    vector: Sequence[ScoredDoc],
    alpha: float = 0.5,
) -> list[ScoredDoc]:
    """Fusion lineaire ponderee avec normalisation min-max de chaque liste."""

    def _norm(lst: Sequence[ScoredDoc]) -> dict[str, float]:
        if not lst:
            return {}
        scores = [d.score for d in lst]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if span <= 0.0:
            return {d.doc_id: 1.0 for d in lst}
        return {d.doc_id: (d.score - lo) / span for d in lst}

    nb = _norm(bm25)
    nv = _norm(vector)
    bm25_ids = {d.doc_id for d in bm25}
    vec_ids = {d.doc_id for d in vector}
    ids = set(nb) | set(nv)
    fused = [
        ScoredDoc(
            doc_id=did,
            score=alpha * nb.get(did, 0.0) + (1.0 - alpha) * nv.get(did, 0.0),
        )
        for did in ids
    ]
    fused.sort(key=lambda d: d.score, reverse=True)
    for i, doc in enumerate(fused):
        doc.rank = i + 1
        in_b = doc.doc_id in bm25_ids
        in_v = doc.doc_id in vec_ids
        doc.source = (
            "bm25+vector" if in_b and in_v else ("bm25" if in_b else "vector")
        )
    return fused


def fuse_hybrid(
    bm25: Sequence[ScoredDoc],
    vector: Sequence[ScoredDoc],
    mode: str = "rrf",
    alpha: float = 0.5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[ScoredDoc]:
    """Fonction commode : fusionne BM25 et vecteurs en un appel."""
    return HybridFuser(mode=mode, alpha=alpha, rrf_k=rrf_k).fuse(bm25, vector)


__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_RRF_K",
    "FR_MODELS",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "SentenceTransformersProvider",
    "OllamaProvider",
    "Embeddings",
    "ScoredDoc",
    "HybridFuser",
    "content_hash",
    "to_dim",
    "l2_normalize",
    "_rrf",
    "_weighted",
    "fuse_hybrid",
]