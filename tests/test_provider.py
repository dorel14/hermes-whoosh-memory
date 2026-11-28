"""Fournisseur de mémoire, testé contre le vrai ``agent.memory_provider`` de Hermes."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path

import pytest

if os.environ.get("CI"):  # en CI le contrat est obligatoire : pas de skip silencieux
    import agent.memory_provider as memory_provider
else:
    memory_provider = pytest.importorskip("agent.memory_provider")

ROOT = Path(__file__).resolve().parents[1]


class FakeCtx:
    def __init__(self):
        self.providers = []

    def register_memory_provider(self, provider):
        self.providers.append(provider)


def _load_plugin_like_hermes():
    """Charge la racine du dépôt comme Hermes charge un plugin répertoire :
    un package au nom synthétique, imports relatifs compris."""
    import types

    sys.modules.setdefault("hermes_user_plugins", types.ModuleType("hermes_user_plugins"))
    sys.modules["hermes_user_plugins"].__path__ = []  # namespace synthétique, comme Hermes
    name = "hermes_user_plugins.whoosh_memory__source_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def plugin():
    return _load_plugin_like_hermes()


@pytest.fixture
def provider(plugin, tmp_path):
    ctx = FakeCtx()
    plugin.register(ctx)
    p = ctx.providers[0]
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli", agent_context="primary")
    yield p
    p.shutdown()


def _call(p, tool, **args):
    return json.loads(p.handle_tool_call(tool, args))


def test_register_gives_a_new_provider_each_time(plugin):
    ctx = FakeCtx()
    plugin.register(ctx)
    plugin.register(ctx)
    a, b = ctx.providers
    assert a is not b
    assert isinstance(a, memory_provider.MemoryProvider)
    assert a.name == "whoosh-memory"


def test_overrides_keep_the_base_signatures(plugin):
    cls = plugin.WhooshMemoryProvider
    base = memory_provider.MemoryProvider
    for name, member in vars(cls).items():
        if name.startswith("_") or not callable(member) or not hasattr(base, name):
            continue
        ours = list(inspect.signature(member).parameters)
        theirs = list(inspect.signature(getattr(base, name)).parameters)
        assert ours == theirs, f"{name}: {ours} != {theirs}"


def test_hooks_in_manifest_are_implemented(plugin):
    import re

    text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    hooks = re.findall(r"^\s+-\s+(\w+)\s*$", text.split("hooks:", 1)[1], re.M)
    assert hooks
    for hook in hooks:
        assert hook in vars(plugin.WhooshMemoryProvider), hook


def test_turn_is_indexed_then_recalled(provider):
    provider.sync_turn("Mon serveur Proxmox s'appelle optiplex", "Noté.", session_id="s1")
    provider.on_session_end([])  # attend la fin des écritures
    context = provider.prefetch("comment s'appelle mon serveur Proxmox ?")
    assert "optiplex" in context
    assert provider.recall_status().count >= 1


def test_trivial_prompt_skips_recall(provider):
    assert provider.prefetch("ok") == ""
    assert provider.recall_status() is None


def test_sync_turn_does_not_block(provider):
    start = time.perf_counter()
    for i in range(20):
        provider.sync_turn(f"question {i}", f"réponse {i}")
    assert time.perf_counter() - start < 0.5
    provider.on_session_end([])
    assert _call(provider, "whoosh_stats")["documents"] == 20


@pytest.mark.parametrize("agent_context", ["subagent", "cron", "flush"])
def test_no_writes_outside_primary_context(plugin, tmp_path, agent_context):
    p = plugin.WhooshMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path), platform="cli", agent_context=agent_context)
    p.sync_turn("secret du cron", "ok")
    p.on_memory_write("add", "memory", "entrée")
    p.on_session_end([])
    assert _call(p, "whoosh_stats")["documents"] == 0
    p.shutdown()


def test_memory_write_add_replace_remove(provider):
    provider.on_memory_write("add", "user", "Préfère le français")
    provider.on_memory_write(
        "replace", "user", "Préfère le français et les réponses courtes",
        {"previous_content": "Préfère le français"},
    )
    provider.on_session_end([])
    assert _call(provider, "whoosh_stats")["by_type"] == {"memory": 1}
    provider.on_memory_write(
        "remove", "user", "", {"previous_content": "Préfère le français et les réponses courtes"}
    )
    provider.on_session_end([])
    assert _call(provider, "whoosh_stats")["documents"] == 0


def test_tools_round_trip(provider):
    added = _call(provider, "whoosh_add", title="Matériel", content="Le NAS est un Synology")
    assert added["ok"] and added["id"].startswith("explicit:")
    hits = _call(provider, "whoosh_search", query="Synology")["results"]
    assert hits[0]["doc_id"] == added["id"]
    assert "Synology" in _call(provider, "whoosh_open", id=added["id"])["content"]
    assert _call(provider, "whoosh_forget", id=added["id"]) == {"removed": 1}


def test_tool_errors_are_json(provider):
    assert "error" in _call(provider, "outil_inconnu")
    assert "error" in _call(provider, "whoosh_search")
    assert "error" in _call(provider, "whoosh_open", id="absent")


def test_tool_names_match_schemas(provider):
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert names == {"whoosh_search", "whoosh_open", "whoosh_add", "whoosh_forget", "whoosh_stats"}
    for schema in provider.get_tool_schemas():
        assert "index_dir" not in schema["parameters"]["properties"]


def test_profiles_are_isolated(plugin, tmp_path):
    a, b = plugin.WhooshMemoryProvider(), plugin.WhooshMemoryProvider()
    a.initialize("s", hermes_home=str(tmp_path / "a"), platform="cli")
    b.initialize("s", hermes_home=str(tmp_path / "b"), platform="cli")
    _call(a, "whoosh_add", content="seulement dans A")
    assert _call(b, "whoosh_search", query="seulement")["results"] == []
    assert (tmp_path / "a" / "whoosh-memory" / "index").is_dir()
    a.shutdown()
    b.shutdown()


def test_save_config_merges_flat_json(provider, tmp_path):
    provider.save_config({"prefetch_top_k": 5}, str(tmp_path))
    provider.save_config({"index_sessions": False}, str(tmp_path))
    data = json.loads((tmp_path / "whoosh-memory" / "config.json").read_text())
    assert data == {"prefetch_top_k": 5, "index_sessions": False}
