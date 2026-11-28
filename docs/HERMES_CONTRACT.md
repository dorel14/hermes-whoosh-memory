# HERMES_CONTRACT.md

Contrat de `MemoryProvider` et de l'enregistrement mémoire, relevé à la source dans le code Hermes installé.

## Source de vérité

- `/opt/hermes/agent/memory_provider.py` (Hermes Agent v0.23.4, build 2026.9.24)
- `/opt/hermes/hermes_cli/plugins_manifest.py` (champs du manifeste connus)

## 1. `register_memory_provider`

- `ctx.register_memory_provider(provider)` — `hermes_cli/plugin_context.py:545`
- Le fournisseur doit être une instance de `MemoryProvider` (ou sous-classe).
- `register(ctx)` reçoit un `ctx` qui expose `register_memory_provider`.
- Chaque appel à `register()` doit créer une **nouvelle instance** (jamais de singleton) — voir le contrat de multiplexage dans `agent/memory_provider.py:84-107`.

## 2. `MemoryProvider` — signatures exactes (avec numéros de ligne)

Toutes les signatures sont copiées de `/opt/hermes/agent/memory_provider.py`.

### Attribut de classe

- `pre_compress_checkpoint_api_version = 1` (ligne 86) — passe à `PRE_COMPRESS_CHECKPOINT_API_VERSION` (valeur 2, ligne 37) si le fournisseur checkpointe durablement chaque `on_pre_compress()`.

### Propriété abstraite

- `name` → `str` (propriété, `@abstractmethod`, lignes 109-111)
  > Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight').

### Méthodes abstraites (à implémenter obligatoirement)

- `is_available(self) -> bool` (@abstractmethod, ligne 117)
  > Configured, credentialed and ready? Gates activation; check config/deps only, no network.
- `initialize(self, session_id: str, **kwargs) -> None` (@abstractmethod, ligne 121)
  > `kwargs` inclut toujours `hermes_home` (chemin de stockage du profil, ne jamais durcir `~/.hermes`) et `platform`; peut inclure `agent_context` ("primary" | "subagent" | "cron" | "flush" — pas d'écriture pour les contextes non primaires), `agent_identity`, `agent_workspace`, `parent_session_id`, `user_id`, `user_id_alt`. Les builds agent passent aussi `resumed` (bool) et `message_count` (int).
- `get_tool_schemas(self) -> list[dict[str, Any]]` (@abstractmethod, ligne 166)
  > OpenAI function-calling schemas ({"name", "description", "parameters"}); [] if none.

### Méthodes concrètes avec implémentation par défaut (à surcharger si besoin)

- `unavailable_reason(self) -> str` (ligne 136) → `""`
- `system_prompt_block(self) -> str` (ligne 140) → `""`
- `prefetch(self, query: str, *, session_id: str = "") -> str` (ligne 144) → `""`
  > Formatted recall context for the upcoming turn ("" if none). Must be fast — recall in the background and return cached results; session_id scopes concurrent sessions.
- `queue_prefetch(self, query: str, *, session_id: str = "") -> None` (ligne 149) → `None`
- `recall_status(self) -> Optional[RecallStatus]` (ligne 152) → `None`
- `sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages: Optional[list[dict[str, Any]]] = None, turn_author: Optional[dict[str, Any]] = None) -> None` (ligne 157) → `None`
- `handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str` (ligne 169) → lève `NotImplementedError`
  > Handle one of this provider's tools; must return a JSON string.
- `shutdown(self) -> None` (ligne 173) → `None`
- `on_turn_start(self, turn_number: int, message: str, **kwargs) -> None` (ligne 178) → `None`
- `identity_signature(self) -> dict[str, Any]` (ligne 191) → `{}`
- `on_session_end(self, messages: list[dict[str, Any]]) -> None` (ligne 197) → `None`
- `on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, rewound: bool = False, **kwargs) -> None` (ligne 200) → `None`
- `on_pre_compress(self, messages: list[dict[str, Any]]) -> str` (ligne 208) → `""`
- `on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None` (ligne 212) → `None`
- `get_config_schema(self) -> list[dict[str, Any]]` (ligne 215) → `[]`
  > Champs pour `hermes memory setup`: key, description, optional secret (va dans .env), required, default, choices, type (text | integer | number | boolean), minimum/maximum/step, url, env_var.
- `save_config(self, values: dict[str, Any], hermes_home: str) -> None` (ligne 222) → `None`
  > Plugins MUST either override this or use only env vars.
- `on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None` (ligne 226) → `None`
  > action: add | replace | remove; target: memory | user; metadata: provenance.
- `backup_paths(self) -> list[str]` (ligne 236) → `[]`
- `on_profile_created(self, profile_name: str, hermes_home: str) -> None` (ligne 245) → `None`
- `on_profile_renamed(self, old_name: str, new_name: str, hermes_home: str) -> None` (ligne 250) → `None`
- `read_config_fields(self, hermes_home: str) -> dict[str, Any]` (ligne 256) → `{"values": {}, "placeholders": {}}`
- `write_config_fields(self, values: dict[str, Any], hermes_home: str) -> None` (ligne 262) → lève `NotImplementedError`

### Helpers de module

- `spawn_context_thread(target: Callable[..., Any], *, name: str, daemon: bool = True, args: tuple = (), kwargs: Optional[dict[str, Any]] = None) -> threading.Thread` (ligne 29)
  > Unstarted thread running target under the spawner's contextvars. Every memory-provider background job must go through this. Jamais `threading.Thread` directement.
- `ctx_bound(fn: Callable[..., Any]) -> Callable[..., Any]` (ligne 21)
- `RecallStatus` dataclass (ligne 53): `provider_label: str`, `count: int`, `glyph: str = INDICATOR_GLYPH`
- `is_core_memory_provider(name: Optional[str]) -> bool` (ligne 47)
- `is_trivial_prompt(text: Optional[str]) -> bool` (ligne 75)

## 3. Champs de manifeste reconnus (`PluginManifest`)

Source: `/opt/hermes/hermes_cli/plugins_manifest.py`, `_KNOWN_MANIFEST_FIELDS` (lignes 33-38):

```
name, version, description, author, requires_env, provides_tools, provides_hooks,
kind, hooks, label, optional_env, platforms, external_dependencies,
pip_dependencies, provides_browser_providers, provides_web_providers,
manifest_version, api_version, requires_plugins, python_dependencies,
config_schema, license, homepage, tags, capabilities, emits, listens, hermes,
depends, requires_hermes, python_runtime, provides_locales, requires_auth, oauth,
config_keys
```

- `requires_hermes`: SemVer floor, ex. `">=0.21"`.
- `manifest_version`: 2 recommandé.
- `api_version`: 1.
- `version`: chaîne entre guillemets (ex. `"0.4.0"`).
- `python_dependencies` : `plugins_manifest.py` ne fait que le valider. L'installation est faite par
  le gestionnaire de paquets (`pm/plugin_declarations.py:read_python_declaration`) après consentement
  (`hermes_cli/plugins_cmd_install.py:_install_plugin_python_deps`), au moment où l'activation est admise.
  Un `pyproject.toml` avec une table `[project]` est prioritaire sur le manifeste. S'il contient
  `[build-system]` ou des `optional-dependencies` non vides, le plugin est construit et installé
  comme un package : ce dépôt n'en déclare donc pas.

## 4. Type de plugin

Ce dépôt est un **fournisseur de mémoire** (`register_memory_provider`, classe `MemoryProvider`), PAS un plugin général (`register_tool` / `register_hook` avec `prefetch`, `sync_turn`, etc.).

`prefetch`, `sync_turn`, `on_memory_write`, `on_pre_compress`, `on_session_end` sont des **méthodes de `MemoryProvider`**, pas des hooks.

## 5. Chemin de stockage

Tous les chemins dérivent de `hermes_home` reçu dans `initialize()`. Jamais `~/.hermes`, jamais `/opt/data`, jamais un chemin fourni par le modèle. Le dossier d'index par défaut: `hermes_home / "whoosh-memory" / "index"`.
## 6. Découverte et chargement d'un fournisseur répertoire

Source : `plugins/memory/__init__.py` (dépôt hermes-agent, commit 374a44e1).

- Emplacement : `$HERMES_HOME/plugins/<nom>/`, avec `__init__.py` **à la racine** du dossier.
- `_is_memory_provider_dir` : le texte de ce `__init__.py` doit mentionner `register_memory_provider`
  ou `MemoryProvider` (sinon une analyse AST des `.py` de premier niveau).
- `_load_provider_from_dir` charge le dossier comme un package sous un nom synthétique
  (`<namespace>.<nom>__source_<hash>`) : les imports relatifs fonctionnent, les imports absolus vers
  un sous-dossier du plugin ne fonctionnent pas.
- `register(ctx)` est appelé en premier ; à défaut, la première sous-classe de `MemoryProvider` trouvée.
- Configuration par défaut (`STORAGE_FLAT_JSON`) : `$HERMES_HOME/<nom>/config.json`.
- `backup_paths()` ne concerne que l'état **hors** de `HERMES_HOME` (ici : rien).
