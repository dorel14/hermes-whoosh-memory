"""Tests pour le module search (R7 : _tokenize)."""
from __future__ import annotations

import sys
import unittest

sys.path.insert(0, ".")

from whoosh_memory.search import _tokenize


class TestTokenize(unittest.TestCase):
    """_tokenize doit produire des tokens alphanumériques en minuscules."""

    def test_simple_tokens(self):
        self.assertEqual(_tokenize("Bonjour le monde"), ["bonjour", "le", "monde"])

    def test_ignores_punctuation(self):
        self.assertEqual(
            _tokenize("Hello, world! 123-test."),
            ["hello", "world", "123", "test"],
        )

    def test_accents_preserved(self):
        self.assertEqual(_tokenize("crème brûlée"), ["crème", "brûlée"])

    def test_empty_string(self):
        self.assertEqual(_tokenize(""), [])

    def test_only_punctuation(self):
        self.assertEqual(_tokenize("!!! ???"), [])


if __name__ == "__main__":
    unittest.main()