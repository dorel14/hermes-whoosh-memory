"""whoosh-memory plugin for Hermes Agent — package entry point.

Local hybrid (BM25 + vector) memory provider: indexes Hermes session
history, MEMORY.md / USER.md, local documents, skills and images into a
whoosh-ng index for fast offline retrieval with automatic context
prefetch before each turn. 100 % local, no network calls.

The real plugin registration lives in ``whoosh_memory/__init__.py``; this
thin shim re-exports everything so ``import whoosh_memory`` works whether
the package is installed via pip or added directly to PYTHONPATH.
"""

from whoosh_memory import (
    DEFAULTS,
    load_config,
    register,
    __version__,
)

__all__ = ["__version__", "register", "DEFAULTS", "load_config"]
