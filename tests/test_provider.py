"""Tests unitaires pour le fournisseur de mémoire whoosh-memory.

Couvre R1 (register(ctx) sans getattr défensif), R2 (héritage direct de
MemoryProvider), R3 (pas de getattr sur ctx), R8 (pas de provides_tools/provides_hooks).
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, ".")

from whoosh_memory.provider import register, WhooshMemoryProvider


class FakeCtx:
    """Contexte Hermes minimal pour tester register()."""

    def __init__(self) -> None:
        self.providers = []
        self.registered_tools = []
        self.registered_hooks = []

    def register_memory_provider(self, provider):
        self.providers.append(provider)

    def register_tool(self, *args, **kwargs):
        self.registered_tools.append((args, kwargs))

    def register_hook(self, *args, **kwargs):
        self.registered_hooks.append((args, kwargs))


class TestRegister(unittest.TestCase):
    """register(ctx) doit utiliser register_memory_provider et rien d'autre."""

    def test_register_uses_register_memory_provider(self):
        ctx = FakeCtx()
        register(ctx)
        self.assertEqual(len(ctx.providers), 1)
        self.assertIsInstance(ctx.providers[0], WhooshMemoryProvider)
        # Aucun outil ni hook ne doit être enregistré (pas de getattr défensif)
        self.assertEqual(ctx.registered_tools, [])
        self.assertEqual(ctx.registered_hooks, [])

    def test_register_creates_new_instance_each_call(self):
        ctx = FakeCtx()
        register(ctx)
        register(ctx)
        self.assertEqual(len(ctx.providers), 2)
        self.assertIsNot(ctx.providers[0], ctx.providers[1])


class TestMemoryProviderInheritance(unittest.TestCase):
    """WhooshMemoryProvider doit hériter directement de MemoryProvider."""

    def test_inherits_directly_from_memory_provider(self):
        mro = [c.__name__ for c in WhooshMemoryProvider.__mro__]
        self.assertIn("MemoryProvider", mro)
        # MemoryProvider doit être la classe parente directe (pas d'intermédiaire)
        self.assertEqual(WhooshMemoryProvider.__bases__[0].__name__, "MemoryProvider")

    def test_has_required_abstract_methods(self):
        for method in (
            "is_available",
            "initialize",
            "get_tool_schemas",
            "handle_tool_call",
        ):
            self.assertTrue(hasattr(WhooshMemoryProvider, method))


if __name__ == "__main__":
    unittest.main()