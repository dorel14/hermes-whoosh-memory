"""Moteur de whoosh-memory : index whoosh-ng local (BM25, vecteurs optionnels).

Ce sous-package ne dépend pas de Hermes. Le fournisseur de mémoire
(``provider.WhooshMemoryProvider``) est chargé par le ``__init__.py`` à la
racine du plugin, qui est le point d'entrée lu par Hermes.
"""

from .config import DEFAULTS, load_config
from .search import SearchHit, index_documents, whoosh_search
from .tools import whoosh_add, whoosh_forget, whoosh_open, whoosh_stats

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "DEFAULTS",
    "load_config",
    "SearchHit",
    "whoosh_search",
    "index_documents",
    "whoosh_open",
    "whoosh_forget",
    "whoosh_stats",
    "whoosh_add",
]
