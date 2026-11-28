# whoosh-memory — mémoire locale pour Hermes Agent

Fournisseur de mémoire (`MemoryProvider`) pour [Hermes Agent](https://github.com/NousResearch/hermes-agent),
basé sur [whoosh-ng](https://github.com/dorel14/whoosh-ng). Il indexe localement :

- chaque tour de conversation (`sync_turn`) ;
- les écritures de la mémoire intégrée, MEMORY.md et USER.md (`on_memory_write`, avec ajout, remplacement et suppression) ;
- les notes que l'agent enregistre explicitement (outil `whoosh_add`).

Avant chaque tour, il injecte les souvenirs pertinents (`prefetch`). Tout est 100 % local, sans appel réseau.

## Installation

```bash
hermes plugins install dorel14/hermes-whoosh-memory --enable
```

Hermes installe le plugin dans `$HERMES_HOME/plugins/whoosh-memory/`, puis demande l'accord pour
installer sa dépendance Python (`whoosh-ng>=5.4,<6`, déclarée dans `pyproject.toml`). En mode non
interactif, ajoute `--yes-deps`.

Active-le ensuite comme fournisseur de mémoire (un seul fournisseur externe peut être actif à la fois) :

```yaml
# $HERMES_HOME/config.yaml
memory:
  provider: whoosh-memory
```

puis `hermes memory setup` pour régler les options.

## Outils exposés à l'agent

| Outil | Rôle |
|-------|------|
| `whoosh_search(query, limit=5)` | Recherche plein texte (BM25) |
| `whoosh_open(id \| path, offset, max_chars)` | Lit un document, page par page |
| `whoosh_add(content, title?)` | Enregistre une note (l'id est généré) |
| `whoosh_forget(id \| path)` | Supprime un document |
| `whoosh_stats()` | Nombre de documents par type |

Chaque outil renvoie du JSON, y compris en cas d'erreur (`{"error": "..."}`).

## Configuration

Fichier : `$HERMES_HOME/whoosh-memory/config.json`, modifiable via `hermes memory setup` ou le panneau
mémoire du dashboard.

| Clé | Défaut | Rôle |
|-----|--------|------|
| `index_sessions` | `true` | Indexer chaque tour de conversation |
| `prefetch_top_k` | `3` | Nombre de souvenirs injectés avant un tour |
| `prefetch_max_chars` | `3000` | Budget de caractères du contexte injecté |

L'index est stocké dans `$HERMES_HOME/whoosh-memory/index/`, un index par profil Hermes.

## Comportement

- **Sans blocage.** Les écritures passent par une file traitée par un thread unique (`spawn_context_thread`).
- **Contextes secondaires.** Les sessions `cron`, `subagent` et `flush` n'écrivent rien.
- **Mode dégradé.** Sans whoosh-ng, le plugin bascule sur un BM25 en pur Python (`index_bm25.json`).
- **Vie privée.** Aucun dossier personnel (documents, photos) n'est indexé par défaut.

## Dépendances optionnelles

Les modules d'embeddings (`embeddings.py`) et d'images (`images.py`) ne sont pas branchés sur le
fournisseur dans cette version. Ils s'importent sans leurs dépendances (`fastembed`, `Pillow`…), qui
s'installent à la main si besoin.

## Développement

```bash
pip install whoosh-ng pytest
# contrat Hermes réel (module 100 % bibliothèque standard)
mkdir -p .hermes-contract/agent && touch .hermes-contract/agent/__init__.py
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/agent/memory_provider.py \
  -o .hermes-contract/agent/memory_provider.py
pytest
```

Le contrat relevé dans le code source de Hermes est documenté dans [docs/HERMES_CONTRACT.md](docs/HERMES_CONTRACT.md).

## Licence

MIT, © dorel14.
