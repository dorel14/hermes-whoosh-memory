from __future__ import annotations

"""Package whoosh-memory (plugin Hermes).

Fournit un backend de mémoire local hybride (BM25 + embeddings) indexant
l'historique des sessions, MEMORY.md / USER.md, les documents locaux, les
skills Hermes et les images dans un index whoosh-ng pour une récupération
hors-ligne rapide avec prefetch automatique avant chaque turn.

Le plugin est 100 % local : aucun appel réseau n'est nécessaire. La recherche
vectorielle utilise le champ natif ``whoosh_modern.fields.VECTOR`` (v0.3.0).

C'est un Memory Provider Hermes : l'entrée d'enregistrement est
``register(ctx)``, qui appelle ``ctx.register_memory_provider(provider)``
(siège dans :mod:`whoosh_memory.provider`).
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
from .provider import WhooshMemoryProvider, register
from .search import SearchHit, index_documents, whoosh_search
from .tools import whoosh_add, whoosh_forget, whoosh_open, whoosh_stats

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "register",
    "WhooshMemoryProvider",
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