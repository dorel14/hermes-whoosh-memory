from __future__ import annotations

"""Package whoosh-memory (plugin Hermes).

Fournit un backend de mémoire local hybride (BM25 + embeddings) indexant
l'historique des sessions, MEMORY.md / USER.md, les documents locaux et les
skills Hermes dans un index whoosh-ng pour une récupération hors-ligne rapide.

Le plugin est 100 % local : aucun appel réseau n'est nécessaire. La recherche
vectorielle utilise le champ natif ``whoosh_modern.fields.VECTOR`` (v0.3.0).
"""

from .config import DEFAULTS, load_config

__version__ = "0.3.0"

__all__ = ["__version__", "register", "DEFAULTS", "load_config"]


def register(ctx) -> None:
    """Point d'entrée du plugin (appelé par Hermes au chargement).

    Enregistre les outils, hooks et commandes fournis par whoosh-memory.

    Args:
        ctx: contexte du plugin fourni par Hermes (tools, hooks, log, config...).
    """
    # Scaffold : l'enregistrement effectif des outils/hooks sera branché ici.
    return None
