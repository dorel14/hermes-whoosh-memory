# whoosh-memory — Mémoire locale hybride pour Hermes Agent

Un **Memory Provider** Hermes 100 % local qui indexe l'historique des sessions,
MEMORY.md / USER.md, les documents locaux, les skills Hermes et les images dans
un index **whoosh-ng (BM25 + recherche vectorielle native)** pour une
récupération hors-ligne rapide avec **prefetch automatique** avant chaque turn.

- 100 % local : aucun appel réseau n'est obligatoire (Ollama en option).
- Dégradé proprement en **BM25 pur** si aucun provider d'embeddings n'est
  disponible.
- Importable sans whoosh-ng : le backend whoosh-ng est chargé paresseusement
  par `indexer.py` / `search.py`.

## Enregistrement (Memory Provider)

Le plugin s'enregistre via `register(ctx)` :

```python
# whoosh_memory/provider.py
def register(ctx) -> None:
    ctx.register_memory_provider(WhooshMemoryProvider())
```

`WhooshMemoryProvider` hérite de `agent.memory_provider.MemoryProvider` et
implémente `initialize`, `unavailable_reason`, `system_prompt_block`,
`prefetch`, `sync_turn`, `on_memory_write`, `read_config_fields`,
`write_config_fields`, `backup_paths`. Aucun `provides_tools` /
`provides_hooks` dans `plugin.yaml` : ce sont des méthodes du provider, pas des
hooks de plugin générique.

## Gestion vectorielle (whoosh-ng moderne)

Depuis la v0.3.0, la recherche vectorielle utilise le **vrai champ vectoriel
natif de whoosh-ng**, `whoosh_modern.fields.VECTOR` — et non l'API 2.x
classique (`whoosh.fields.VECTOR` n'existe pas ; `VectorField(dimensions=...)`
de `whoosh.vector` est une API distincte, à fournisseur externe).

- `schema.get_schema(dimension=384)` déclare `embedding=VECTOR(stored=True,
  dimension=dimension, normalized=True)` : chaque vecteur est L2-normalisé à
  l'écriture, donc la recherche par `VectorSearcher` est une **similarité
  cosinus exacte** (produit scalaire).
- `search.whoosh_search(..., mode="vector")` → `VectorSearcher` (cosinus pur,
  top-k via `np.argpartition`).
- `search.whoosh_search(..., mode="hybrid")` → `HybridSearcher` en **mode
  colonne native** (`vector_column="embedding"`), sans provider externe : le
  vecteur vit dans l'index, la fusion se fait par `docnum` (pondérée via
  `alpha`, ou RRF via `rrf_k`).
- `indexer.index_document(..., embedding=vec)` stocke le vecteur d'un document
  de l'index principal ; `search.index_documents(..., embeddings=...)` encode
  chaque chunk à l'indexation.
- La dimension du champ doit correspondre à celle du provider d'embeddings :
  `VECTOR(dimension=...)` valide la longueur à l'écriture (sinon
  `ValueError`). Passez `dimension=` à `whoosh_search`/`index_documents`.

Mode effectif résolu par `whoosh_search` : `mode` (`bm25` | `vector` |
`hybrid` | `auto`) est prioritaire ; sinon `hybrid` (bool/str) puis
`use_embeddings`. Sans provider d'embeddings disponible, `vector`/`hybrid`
retombent sur BM25.

## Installation

### Depuis les sources (développement)

```bash
git clone https://github.com/dorel14/hermes-whoosh-memory.git
cd hermes-whoosh-memory
pip install -e .
```

### Depuis GitHub (utilisateur)

```bash
pip install git+https://github.com/dorel14/hermes-whoosh-memory.git
```

### Dépendances

- Requis : `whoosh-ng>=0.3.0`
- Optionnel : `fastembed` (CPU/ONNX, recommandé), `sentence-transformers`
  (PyTorch), `ollama`, `Pillow` (images)

```bash
pip install "hermes-whoosh-memory[all]"
```

## Configuration

```yaml
# HERMES_HOME/plugin-data/whoosh-memory/config.yaml
whoosh_memory:
  index_dir: "plugin-data/whoosh-memory/index"
  skills_index_dir: "plugin-data/whoosh-memory/skills-index"
  default_top_k: 10
  hybrid_enabled: "auto"          # auto | true | false
  hybrid_method: "rrf"            # rrf | weighted
  hybrid_weight_bm25: 0.65
  hybrid_weight_vector: 0.35
  embedding_provider: "auto"      # auto | fastembed | sentence-transformers | ollama
  embedding_model: "multilingual-e5-small"
  embedding_dim: 384
  embedding_cache_dir: "plugin-data/whoosh-memory/embeddings_cache"
  chunk_size: 1000
  chunk_overlap: 150
  index_images: true
  index_image_paths: ["~/Photos", "~/Pictures", "~/screenshots"]
  image_ocr: false
  image_embedding: true
  image_embedding_provider: "libembedding-ng"
  image_max_mb: 20
  image_max_dimension: 1024
  image_base64_max_mb: 10
```

Tous les chemins dérivent de `HERMES_HOME` reçu dans `initialize()`. Aucun
chemin en dur (`~/.hermes`, `/opt/data`, chemin fourni par un modèle).

## Utilisation CLI

```bash
whoosh-memory init              # créer les dossiers + config par défaut
whoosh-memory config            # afficher la configuration
whoosh-memory search "requête"  # rechercher dans l'index
whoosh-memory index ./docs      # indexer un dossier de documents
whoosh-memory index-images ./Photos  # indexer un dossier d'images
whoosh-memory stats             # statistiques de l'index
```

## Providers et modèles français

| Provider | Modèles | Dimension | Notes |
|---|---|---|---|
| fastembed | multilingual-e5-small | 384 | CPU ONNX, sans PyTorch |
| fastembed | multilingual-e5-base | 768 | CPU ONNX, sans PyTorch |
| fastembed | bge-small | 512 | CPU ONNX, sans PyTorch |
| sentence-transformers | BAAI/bge-m3 | 1024 | PyTorch |
| sentence-transformers | multilingual-e5-base | 768 | PyTorch |
| ollama | nomic-embed-text | 768 | requiert un serveur Ollama local |

## Fusion hybride

Deux stratégies, pilotées par `hybrid_method` (config) ou le paramètre `mode`
de `fuse_hybrid` :

- **RRF** (`mode="rrf"`) : `score = sum(1 / (k + rang))`, k=60 par défaut.
  Idéal quand BM25 et les vecteurs ont des échelles de score incomparables.
- **Pondérée** (`mode="weighted"`) : `score = alpha * bm25_norm + (1 - alpha) * vector_norm`,
  avec normalisation min-max. `alpha = hybrid_weight_bm25` par défaut (0.65).

## Cache par haché de contenu

Chaque document est haché (SHA-256) à l'indexation ; un document est réindexé
uniquement si son haché a changé (`needs_reindex`).

## Mode dégradé (BM25 pur)

Sans provider d'embeddings disponible :

- la recherche vectorielle (`vector`) et la fusion (`hybrid`) retombent
  automatiquement sur BM25 ;
- `HybridFuser` avec `mode="bm25"` retourne simplement les résultats BM25.

## Sécurité

Les chemins sous `HERMES_HOME` contenant des credentials sont exclus par une
denylist stricte. Aucun identifiant, clé d'API ou jeton MCP n'est stocké en
clair : ils sont anonymisés par `[REDACTED]`. Les images sont indexées avec
chemin absolu + métadonnées + embedding optionnel (jamais le base64 ni le
binaire stocké) ; la conversion base64 se fait à la volée via
`whoosh_image_base64`.

## Licence

MIT — voir `LICENSE`.