"""Moteur d'index : les deux backends (whoosh-ng et BM25 pur) se comportent pareil."""

from __future__ import annotations

import importlib
import sys

import pytest

from whoosh_memory import search, tools


@pytest.fixture(params=["whoosh", "bm25_pure"])
def index_dir(request, tmp_path, monkeypatch):
    if request.param == "bm25_pure":
        monkeypatch.setattr(search, "_WHOOSH_AVAILABLE", False)
    d = tmp_path / "index"
    yield str(d), request.param


def test_backend_selected(index_dir):
    d, backend = index_dir
    tools.whoosh_add(d, id="a", path="p", title="t", content="bonjour")
    assert tools.whoosh_stats(d)["backend"] == backend


def test_add_is_an_upsert(index_dir):
    d, _ = index_dir
    tools.whoosh_add(d, id="a", path="p1", title="T", content="le chat dort", type="memory")
    tools.whoosh_add(d, id="a", path="p1", title="T", content="le chien dort", type="memory")
    assert tools.whoosh_stats(d)["documents"] == 1
    assert [h.doc_id for h in search.whoosh_search(d, "chien")] == ["a"]
    assert search.whoosh_search(d, "chat") == []


def test_open_paginates(index_dir):
    d, _ = index_dir
    tools.whoosh_add(d, id="a", path="p1", title="T", content="0123456789")
    doc = tools.whoosh_open(d, id="a", offset=2, max_chars=3)
    assert doc["content"] == "234"
    assert doc["total_chars"] == 10
    assert tools.whoosh_open(d, path="p1")["id"] == "a"
    assert tools.whoosh_open(d, id="absent") is None


def test_forget_by_id_and_path(index_dir):
    d, _ = index_dir
    for i in range(3):
        tools.whoosh_add(d, id=f"d{i}", path="p" if i else "q", title="T", content=f"texte {i}")
    assert tools.whoosh_forget(d, id="d0") == 1
    assert tools.whoosh_forget(d, path="p") == 2
    assert tools.whoosh_stats(d)["documents"] == 0


def test_stats_by_type(index_dir):
    d, _ = index_dir
    tools.whoosh_add(d, id="a", path="p", title="T", content="un", type="memory")
    tools.whoosh_add(d, id="b", path="p", title="T", content="deux", type="session_turn")
    assert tools.whoosh_stats(d)["by_type"] == {"memory": 1, "session_turn": 1}


def test_schemas_build_with_installed_whoosh_ng():
    from whoosh_memory import schema

    schema.get_schema(8)
    search._make_schema(8)


def test_package_imports_without_pillow(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)  # simule Pillow absent
    for name in [m for m in sys.modules if m.startswith("whoosh_memory")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.import_module("whoosh_memory")
    importlib.import_module("whoosh_memory.images")
