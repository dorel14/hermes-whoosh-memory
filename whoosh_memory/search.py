"""Outil whoosh_search (Section 5) — plugin whoosh-memory.

Recherche sur l'index whoosh-ng avec trois modes :

- ``bm25``    : BM25F classique via ``QueryParser`` sur le champ ``content``.
- ``vector``  : recherche vectorielle pure via ``VectorSearcher`` (similarité
  cosinus sur la colonne native ``embedding``).
- ``hybrid``  : fusion BM25 + vecteur via ``HybridSearcher`` en mode colonne
  native (``vector_column="embedding"``) — fusion pondérée (``alpha``) ou RRF.

Quand whoosh-ng n'est pas importable (ou ``numpy`` absent), le plugin se
DÉgrade en BM25 pur Python (``_PyBM25Index``), sans erreur ni appel réseau.

Le mode ``auto`` choisit hybride si un provider d'embeddings est disponible et
que la colonne vectorielle existe, sinon BM25 seul.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _tokenize(text: str) -> list[str]:
    """Tokenise un texte en minuscules (alphanumérique uniquement)."""
    return re.findall(r"[a-zA-Z0-9\u00c0-\u024f]+", text.lower())


from .extractors import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ExtractedDocument,
    chunk_text,
)
from .schema import DEFAULT_EMBEDDING_DIM
from .skills_parser import ParsedSkill

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# whoosh + whoosh_modern sont optionnels (degradation BM25 pur sinon).
try:
    from whoosh.fields import ID, Schema, TEXT  # noqa: F401  (fallback ci-dessous)
    from whoosh.index import create_in, open_dir
    from whoosh.qparser import QueryParser
    from whoosh.scoring import BM25F

    _WHOOSH_AVAILABLE = True
except Exception:  # ImportError / ModuleNotFoundError
    _WHOOSH_AVAILABLE = False

# Composants vectoriels modernes (natif whoosh-ng). Disponibles seulement si
# whoosh_modern + numpy sont importables.
try:
    from whoosh_modern.fields import VECTOR
    from whoosh_modern.searching import HybridSearcher, VectorSearcher

    _VECTOR_AVAILABLE = True
except Exception:  # ImportError / ModuleNotFoundError / numpy absent
    VECTOR = None
    HybridSearcher = None
    VectorSearcher = None
    _VECTOR_AVAILABLE = False


def _make_schema(dimension: int) -> Schema:
    """Schéma de l'index de recherche (doc_id/path/title/content + embedding)."""
    fields = {
        "doc_id": ID(stored=True, unique=True),
        "path": ID(stored=True),
        "title": TEXT(stored=True),
        "content": TEXT(stored=True),
    }
    if _VECTOR_AVAILABLE:
        fields["embedding"] = VECTOR(
            stored=True, dimension=dimension, normalized=True
        )
    return Schema(**fields)


@dataclass
class SearchHit:
    doc_id: str
    path: str
    title: str
    score: float
    snippet: str
    source: str = "bm25"  # "bm25" | "vector" | "hybrid"
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None


class _PyBM25Index:
    """Index BM25 pur en Python (fallback quand whoosh est absent)."""

    k1 = 1.5
    b = 0.75

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def add(self, doc_id: str, path: str, title: str, text: str) -> None:
        tokens = _tokenize(text)
        self.docs.append(
            {
                "id": doc_id,
                "path": path,
                "title": title,
                "text": text,
                "tokens": tokens,
            }
        )
        for token in set(tokens):
            self.df[token] = self.df.get(token, 0) + 1

    def finish(self) -> None:
        n = len(self.docs)
        total = sum(len(d["tokens"]) for d in self.docs)
        self.avgdl = total / n if n else 0.0
        for token, df in self.df.items():
            self.idf[token] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _score(self, tokens: list[str], freqs: dict[str, int], dl: int) -> float:
        if self.avgdl <= 0:
            return 0.0
        score = 0.0
        for token in tokens:
            f = freqs.get(token, 0)
            if f == 0 or token not in self.idf:
                continue
            idf = self.idf[token]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1) / denom)
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.docs or top_k <= 0:
            return []
        qtokens = _tokenize(query)
        if not qtokens:
            return []
        scored: list[tuple[str, float]] = []
        for doc in self.docs:
            freqs: dict[str, int] = {}
            for token in doc["tokens"]:
                freqs[token] = freqs.get(token, 0) + 1
            score = self._score(qtokens, freqs, len(doc["tokens"]) or 1)
            if score > 0:
                scored.append((doc["id"], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _clean(self, word: str) -> str:
        return re.sub(r"\W+", "", word.lower())

    def snippet(self, doc_id: str, query: str, width: int = 220) -> str:
        text = ""
        for doc in self.docs:
            if doc["id"] == doc_id:
                text = doc["text"]
                break
        if not text:
            return ""
        words = text.split()
        if not words:
            return ""
        qtokens = set(_tokenize(query))
        window = max(6, width // 8)
        best_pos, best_count = 0, -1
        for i, word in enumerate(words):
            if self._clean(word) in qtokens:
                lo = max(0, i - window)
                hi = min(len(words), i + window)
                count = sum(1 for w in words[lo:hi] if self._clean(w) in qtokens)
                if count > best_count:
                    best_count, best_pos = count, lo
        take = max(8, width // 8)
        return " ".join(words[best_pos : best_pos + take]).strip()


class _Index:
    """Wrapper : whoosh (+ vectoriel natif) quand disponible, BM25 pur sinon."""

    def __init__(self, index_dir: Path, dimension: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.index_dir = Path(index_dir)
        self.dimension = dimension
        self._whoosh = None
        self._py: Optional[_PyBM25Index] = None
        self._json_path = self.index_dir / "index_bm25.json"
        if _WHOOSH_AVAILABLE:
            try:
                self.index_dir.mkdir(parents=True, exist_ok=True)
                if any(self.index_dir.iterdir()):
                    self._whoosh = open_dir(str(self.index_dir))
                else:
                    self._whoosh = create_in(str(self.index_dir), _make_schema(dimension))
            except Exception:  # index corrompu / verrou : bascule BM25 pur
                self._whoosh = None
                self._py = _PyBM25Index()
                self._load_json()
        else:
            self._py = _PyBM25Index()
            self._load_json()

    def _load_json(self) -> None:
        if not self._json_path.exists():
            return
        try:
            import json

            data = json.loads(self._json_path.read_text(encoding="utf-8"))
        except Exception:
            return
        assert self._py is not None
        self._py.docs = data.get("docs", [])
        self._py.df = data.get("df", {})
        self._py.avgdl = data.get("avgdl", 0.0)

    def _save_json(self) -> None:
        import json

        assert self._py is not None
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        self._json_path.write_text(
            json.dumps(
                {
                    "docs": self._py.docs,
                    "df": self._py.df,
                    "avgdl": self._py.avgdl,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def add(
        self,
        doc_id: str,
        path: str,
        title: str,
        text: str,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        if not text:
            return
        if self._whoosh is not None:
            writer = self._whoosh.writer()
            fields = {"doc_id": doc_id, "path": path, "title": title, "content": text}
            if embedding is not None and _VECTOR_AVAILABLE:
                fields["embedding"] = list(embedding)
            writer.add_document(**fields)
            writer.commit()
        else:
            self._py.add(doc_id, path, title, text)

    def finish(self) -> None:
        if self._py is not None:
            self._py.finish()
            self._save_json()

    # -- BM25 ----------------------------------------------------------------

    def search_bm25(self, query: str, top_k: int) -> list[dict]:
        """Recherche BM25F via QueryParser sur ``content``."""
        if self._whoosh is not None:
            with self._whoosh.searcher(weighting=BM25F) as searcher:
                parser = QueryParser("content", self._whoosh.schema)
                try:
                    q = parser.parse(query)
                except Exception:
                    return []
                results = searcher.search(q, limit=top_k)
                out: list[dict] = []
                for hit in results:
                    out.append(
                        {
                            "doc_id": hit["doc_id"],
                            "path": hit["path"],
                            "title": hit["title"],
                            "score": float(hit.score),
                            "snippet": hit.highlights("content") or "",
                            "bm25_score": float(hit.score),
                            "vector_score": None,
                        }
                    )
                return out
        assert self._py is not None
        ranked = self._py.search(query, top_k)
        meta = {d["id"]: (d["path"], d["title"]) for d in self._py.docs}
        out = []
        for doc_id, score in ranked:
            path, title = meta.get(doc_id, ("", ""))
            out.append(
                {
                    "doc_id": doc_id,
                    "path": path,
                    "title": title,
                    "score": score,
                    "snippet": self._py.snippet(doc_id, query),
                    "bm25_score": score,
                    "vector_score": None,
                }
            )
        return out

    # -- Vectoriel pur --------------------------------------------------------

    def search_vector(self, query_vector: Sequence[float], top_k: int) -> list[dict]:
        """Recherche vectorielle pure via ``VectorSearcher`` (colonne native)."""
        if self._whoosh is None or not _VECTOR_AVAILABLE or VectorSearcher is None:
            return []
        with self._whoosh.searcher() as searcher:
            vs = VectorSearcher(searcher.reader(), "embedding")
            hits = vs.search(list(query_vector), k=top_k)  # [(docnum, score)]
            out: list[dict] = []
            for docnum, score in hits:
                stored = searcher.stored_fields(docnum) or {}
                out.append(
                    {
                        "doc_id": stored.get("doc_id", ""),
                        "path": stored.get("path", ""),
                        "title": stored.get("title", ""),
                        "score": float(score),
                        "snippet": "",
                        "bm25_score": None,
                        "vector_score": float(score),
                    }
                )
            return out

    # -- Hybride (BM25 + vecteur) ---------------------------------------------

    def search_hybrid(
        self,
        query: str,
        query_vector: Sequence[float],
        top_k: int,
        alpha: float = 0.5,
        method: str = "weighted",
        rrf_k: int = 60,
    ) -> list[dict]:
        """Fusion BM25 + vecteur via ``HybridSearcher`` en mode colonne native."""
        if self._whoosh is None or not _VECTOR_AVAILABLE or HybridSearcher is None:
            return []
        with self._whoosh.searcher(weighting=BM25F) as searcher:
            parser = QueryParser("content", self._whoosh.schema)
            try:
                parsed = parser.parse(query)
            except Exception:
                return []
            hybrid = HybridSearcher(
                searcher, id_field="doc_id", vector_column="embedding"
            )
            results = hybrid.search(
                parsed,  # objet Query pre-parsé (évite le défaut schema.names()[0])
                query_vector=list(query_vector),
                limit=top_k,
                alpha=alpha,
                method=method,
                rrf_k=rrf_k,
                vector_k=top_k,
            )
            out: list[dict] = []
            for r in results:
                fields = r.fields or {}
                out.append(
                    {
                        "doc_id": r.doc_id or fields.get("doc_id", ""),
                        "path": fields.get("path", ""),
                        "title": fields.get("title", ""),
                        "score": float(r.score),
                        "snippet": "",
                        "bm25_score": float(r.bm25_score),
                        "vector_score": float(r.vector_score),
                    }
                )
            return out


def _embeddings_available() -> bool:
    for name in ("libembedding_ng", "fastembed", "sentence_transformers"):
        try:
            __import__(name)
            return True
        except Exception:
            continue
    return False


def _resolve_mode(
    mode: str | None,
    hybrid: str | bool,
    use_embeddings: Optional[bool],
    embeddings_ok: bool,
) -> str:
    """Résout le mode effectif (bm25 | vector | hybrid) depuis les paramètres."""
    if mode is not None:
        m = str(mode).strip().lower()
        if m in ("vector", "vec", "embedding", "embeddings"):
            return "vector" if embeddings_ok and _VECTOR_AVAILABLE else "bm25"
        if m in ("hybrid", "mixed", "fusion"):
            return "hybrid" if embeddings_ok and _VECTOR_AVAILABLE else "bm25"
        return "bm25"
    if isinstance(hybrid, bool):
        forced = hybrid
    else:
        forced = str(hybrid).strip().lower() in {"true", "1", "yes", "on"}
    if str(hybrid).strip().lower() in {"false", "0", "no", "off"}:
        return "bm25"
    if forced:
        return "hybrid" if embeddings_ok and _VECTOR_AVAILABLE else "bm25"
    # auto
    if use_embeddings is True:
        return "hybrid" if embeddings_ok and _VECTOR_AVAILABLE else "bm25"
    if use_embeddings is False:
        return "bm25"
    return "hybrid" if embeddings_ok and _VECTOR_AVAILABLE else "bm25"


def whoosh_search(
    index_dir: str | os.PathLike[str],
    query: str,
    *,
    top_k: int = 10,
    hybrid: str | bool = "auto",
    use_embeddings: Optional[bool] = None,
    embeddings=None,
    mode: Optional[str] = None,
    dimension: int = DEFAULT_EMBEDDING_DIM,
    alpha: float = 0.5,
    method: str = "weighted",
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Recherche dans un index whoosh (BM25, vecteur ou hybride).

    Args:
        index_dir: répertoire de l'index whoosh (créé si vide).
        query: texte de la requête.
        top_k: nombre de résultats.
        hybrid: "true"/"false"/"auto" (ou bool) — conservé pour compatibilité.
        use_embeddings: forcé à True/False (compatibilité).
        embeddings: instance ``Embeddings`` (embeddings.py) ; fournit le
            vecteur de requête pour les modes vector/hybrid. Si ``None``,
            la recherche retombe sur BM25 seul.
        mode: "bm25" | "vector" | "hybrid" | "auto" (prioritaire sur hybrid).
        dimension: dimension des vecteurs (doit matcher le schéma d'index).
        alpha: poids BM25 en mode pondéré (vector = 1 - alpha).
        method: "weighted" | "rrf".
        rrf_k: constante k du RRF.

    Returns:
        Liste de SearchHit triée par score décroissant.
    """
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []

    embeddings_ok = bool(embeddings is not None and getattr(embeddings, "available", False))
    mode_eff = _resolve_mode(mode, hybrid, use_embeddings, embeddings_ok)

    index = _Index(Path(index_dir), dimension=dimension)
    index.finish()

    query_vector = None
    if mode_eff in ("vector", "hybrid") and embeddings_ok:
        try:
            query_vector = embeddings.embed(query)
        except Exception:
            query_vector = None
        if query_vector is None:
            mode_eff = "bm25"

    if mode_eff == "vector" and query_vector is not None:
        rows = index.search_vector(query_vector, top_k)
        source = "vector"
    elif mode_eff == "hybrid" and query_vector is not None:
        rows = index.search_hybrid(
            query, query_vector, top_k, alpha=alpha, method=method, rrf_k=rrf_k
        )
        source = "hybrid"
    else:
        rows = index.search_bm25(query, top_k)
        source = "bm25"

    hits: list[SearchHit] = []
    for row in rows:
        hits.append(
            SearchHit(
                doc_id=row["doc_id"],
                path=row["path"],
                title=row["title"],
                score=round(float(row["score"]), 4),
                snippet=(row.get("snippet") or "")[:240],
                source=source,
                vector_score=row.get("vector_score"),
                bm25_score=row.get("bm25_score"),
            )
        )
    return hits


def index_documents(
    index_dir: str | os.PathLike[str],
    documents: Iterable[ExtractedDocument | ParsedSkill],
    *,
    embeddings=None,
    dimension: int = DEFAULT_EMBEDDING_DIM,
) -> int:
    """Indexe des documents (fichiers extraits ou skills parsés).

    Applique le découpage ~1000 caractères / overlap 150. Si ``embeddings``
    est fourni et disponible, chaque chunk est encodé et stocké dans la
    colonne ``embedding`` (recherche vectorielle/hybride ultérieure).
    Retourne le nombre de chunks indexés.
    """
    index = _Index(Path(index_dir), dimension=dimension)
    count = 0
    for document in documents:
        if isinstance(document, ParsedSkill):
            text = document.content
            doc_id = f"skill:{document.path}"
            path = document.path
            title = document.title
        else:
            text = document.text
            doc_id = f"file:{document.path}"
            path = document.path
            title = document.title
        chunks = chunk_text(
            text, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP
        )
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}#{i}" if len(chunks) > 1 else doc_id
            vec = None
            if embeddings is not None and getattr(embeddings, "available", False):
                try:
                    vec = embeddings.embed(chunk)
                except Exception:
                    vec = None
            index.add(chunk_id, path, title, chunk, embedding=vec)
            count += 1
    index.finish()
    return count


__all__ = ["SearchHit", "whoosh_search", "index_documents"]