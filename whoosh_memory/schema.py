"""Schéma d'index whoosh-ng pour le plugin whoosh-memory (spec Section 3).

Champs (tous ``stored``) :
    id, type, source, role, title, content, embedding, timestamp,
    path, session_id, skill_name, metadata, content_hash

Le champ ``embedding`` utilise le VRAI champ vectoriel de whoosh-ng,
``whoosh_modern.fields.VECTOR`` (sous-classe de ``STORED`` dont le
``column_type`` est ``VectorColumn``). C'est lui qui rend l'index exploitable
par ``VectorSearcher`` et ``HybridSearcher`` en mode colonne native.

À noter : le ``whoosh.fields`` legacy (2.x) N'EXPORTE PAS ``VECTOR`` — le
schéma 2.x classique ``VectorField(dimensions=...)`` de ``whoosh.vector``
est une API distincte (fournisseur externe) ; le champ natif moderne est
``whoosh_modern.fields.VECTOR(dimension=..., normalized=...)``.

``normalized=True`` L2-normalise chaque vecteur à l'écriture, ce qui donne
une similarité cosinus exacte via ``VectorSearcher`` (produit scalaire).

Si ``whoosh_modern`` est indisponible (vieille installation whoosh 2.x),
le champ ``embedding`` retombe sur ``STORED`` : le plugin fonctionne alors
en BM25 pur, sans erreur.
"""

from __future__ import annotations

# whoosh est optionnel : le plugin fonctionne en BM25 pur sans lui.
try:
    from whoosh.analysis import StemmingAnalyzer
    from whoosh.fields import DATETIME, ID, KEYWORD, STORED, TEXT, Schema
    _WHOOSH_IMPORT_OK = True
except Exception:  # whoosh non installé -> BM25 pur, pas d'erreur
    StemmingAnalyzer = None
    DATETIME = ID = KEYWORD = STORED = TEXT = Schema = None
    _WHOOSH_IMPORT_OK = False

# Champ vectoriel natif whoosh-ng. Le ``whoosh.fields`` legacy ne l'exporte PAS.
try:
    from whoosh_modern.fields import VECTOR as _MODERN_VECTOR
except Exception:  # whoosh_modern absent (install whoosh 2.x) -> BM25 pur
    _MODERN_VECTOR = None

# Valeurs admises pour le champ ``type``.
TYPE_SESSION_TURN = "session_turn"
TYPE_MEMORY = "memory"
TYPE_FILE = "file"
TYPE_CHUNK = "chunk"
TYPE_SKILL = "skill"
TYPE_EXPLICIT = "explicit"
TYPES = frozenset({
    TYPE_SESSION_TURN, TYPE_MEMORY, TYPE_FILE, TYPE_CHUNK, TYPE_SKILL, TYPE_EXPLICIT,
})

# Valeurs admises pour le champ ``role`` (uniquement pour les turns).
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLES = frozenset({ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL})

DEFAULT_EMBEDDING_DIM = 384


def vector_field_available() -> bool:
    """True si le champ vectoriel natif ``whoosh_modern.fields.VECTOR`` existe."""
    return _MODERN_VECTOR is not None


def get_schema(dimension: int = DEFAULT_EMBEDDING_DIM) -> object | None:
    """Retourne le Schema whoosh-ng de l'index whoosh-memory.

    Args:
        dimension: dimension des vecteurs du champ ``embedding`` (doit
            correspondre à celle du provider d'embeddings configuré).

    Le champ ``embedding`` est un ``VECTOR`` natif (recherche vectorielle
    via ``VectorSearcher``/``HybridSearcher`` en mode colonne native) si
    whoosh-ng moderne est installé, sinon un simple ``STORED`` (BM25 pur).

    Retourne None si whoosh n'est pas installé ; l'appelant doit gérer
    ce cas (search.py utilise _make_schema, indépendant de get_schema).
    """
    if not _WHOOSH_IMPORT_OK or Schema is None:
        return None

    if _MODERN_VECTOR is not None:
        embedding = _MODERN_VECTOR(
            stored=True, dimension=dimension, normalized=True
        )
    else:
        embedding = STORED()

    return Schema(
        id=ID(stored=True, unique=True),
        type=KEYWORD(stored=True),          # session_turn | memory | file | chunk | skill | explicit
        source=ID(stored=True),
        role=KEYWORD(stored=True),          # user | assistant | tool (pour les turns)
        title=TEXT(stored=True),
        content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
        embedding=embedding,                # vecteur natif (similarité cosinus si normalized)
        timestamp=DATETIME(stored=True),
        path=ID(stored=True),
        session_id=ID(stored=True),
        skill_name=ID(stored=True),         # pour les skills
        metadata=STORED,
        content_hash=ID(stored=True),       # SHA-256 (+ mtime/taille) pour l'incrémental
    )


SCHEMA = get_schema()
