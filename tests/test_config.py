"""Tests pour la configuration dynamique des chemins (R6)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from whoosh_memory.config import hermes_home, config_path


class TestHermesHome(unittest.TestCase):
    """hermes_home() doit lever RuntimeError quand $HERMES_HOME est absent."""

    def test_raises_when_hermes_home_unset(self):
        os.environ.pop("HERMES_HOME", None)
        # Recharger le module pour éviter la mise en cache
        import importlib
        import whoosh_memory.config as cfg

        importlib.reload(cfg)
        with self.assertRaises(RuntimeError):
            cfg.hermes_home()

    def test_returns_value_when_set(self):
        os.environ["HERMES_HOME"] = "/tmp/test-hermes-home"
        import importlib
        import whoosh_memory.config as cfg

        importlib.reload(cfg)
        self.assertEqual(cfg.hermes_home(), Path("/tmp/test-hermes-home"))


class TestConfigPath(unittest.TestCase):
    """config_path() doit être lazy (pas de constante de module)."""

    def test_config_path_is_callable(self):
        self.assertTrue(callable(config_path))


if __name__ == "__main__":
    unittest.main()