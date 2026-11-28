"""whoosh-memory — fournisseur de mémoire (MemoryProvider) pour Hermes Agent.

Point d'entrée lu par Hermes quand le plugin est installé dans
``$HERMES_HOME/plugins/whoosh-memory/`` : ``register(ctx)`` appelle
``ctx.register_memory_provider(WhooshMemoryProvider())``.
"""

if __package__:
    from .whoosh_memory.provider import WhooshMemoryProvider, register

    __all__ = ["WhooshMemoryProvider", "register"]
# Sans package parent (pytest qui collecte la racine du dépôt), il n'y a rien à
# exposer : Hermes, lui, charge toujours ce fichier comme un package.
