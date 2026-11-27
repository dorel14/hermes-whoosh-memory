# whoosh-memory — Mémoire hybride BM25 + embeddings locales

Plugin de mémoire pour [Hermes Agent](https://github.com/NousResearch/hermes-agent) :
indexe l'historique des sessions, MEMORY.md / USER.md, les dossiers de documents
locaux et les skills Hermes dans un index **whoosh-ng (BM25)**, avec une couche
semantique optionnelle (**embeddings locaux**) et une **fusion hybride** des
résultats.

- 100 % local : aucun appel réseau n'est obligatoire (Ollama en option).
- Degrade proprement en **BM25 pur** si aucun provider d'embeddings n'est
  disponible.
- Importeble sans whoosh-ng ; le backend whoosh-ng est charge paresseusement
  par `indexer.py` / `search.py`.

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

Les embeddings et la fusion hybride sont specifies aux sections **7.2** et
**7.3** du spec ; le plan de developpement (4 phases) a la section **10**.

## Installation

```bash
# Backend de recherche (obligatoire)
pip install whoosh-ng

# Provider d'embeddings (optionnel, recommande)
pip install fastembed                # CPU / ONNX, sans PyTorch (recommande)
# ou
pip install sentence-transformers    # PyTorch, modeles plus lourds
# ou utilisez Ollama (aucune dependance Python supplementaire)
```

## Configuration

```yaml
# ~/.hermes/plugins/whoosh-memory/config.yaml
whoosh_memory:
  index_dir: "~/.hermes/plugin-data/whoosh-memory/index"
  skills_index_dir: "~/.hermes/plugin-data/whoosh-memory/skills-index"
  default_top_k: 10
  hybrid_enabled: "auto"          # auto | true | false
  hybrid_method: "rrf"            # rrf | weighted
  hybrid_weight_bm25: 0.65
  hybrid_weight_vector: 0.35
  embedding_provider: "auto"      # auto | fastembed | sentence-transformers | ollama
  embedding_model: "multilingual-e5-small"
  embedding_dim: 384
  embedding_cache_dir: "~/.hermes/plugin-data/whoosh-memory/embeddings_cache"
  chunk_size: 1000
  chunk_overlap: 150
```

## Utilisation

```python
from whoosh_memory.embeddings import Embeddings, HybridFuser, ScoredDoc, fuse_hybrid

emb = Embeddings(provider="fastembed", model="multilingual-e5-small")

# 1. Encoder la requete (vecteur de requete)
qvec = emb.embed("comment creer un plugin hermes ?")

# 2. Recherche BM25
bm25 = searcher.search("comment creer un plugin hermes", limit=10)

# 3. Fusion hybride avec une recherche BM25 existante (resultats whoosh)
vector = emb.embed("comment creer un plugin hermes")
fused = fuse_hybrid(bm25, vector, mode="rrf")        # RRF, k=60
fused = fuse_hybrid(bm25, vector, mode="weighted", alpha=0.65)
```

## Providers et modeles francais

| Provider | Modeles | Dimension | Notes |
|---|---|---|---|
| fastembed | multilingual-e5-small | 384 | CPU ONNX, sans PyTorch |
| fastembed | multilingual-e5-base | 768 | CPU ONNX, sans PyTorch |
| fastembed | bge-small | 512 | CPU ONNX, sans PyTorch |
| sentence-transformers | BAAI/bge-m3 | 1024 | PyTorch |
| sentence-transformers | multilingual-e5-base | 768 | PyTorch |
| ollama | nomic-embed-text | 768 | requiert un serveur Ollama local |

## Cache par hache de contenu

Chaque document est hache (SHA-256) a l'indexation ; un document est reindexe
uniquement si sa hache a change (`needs_reindex`).

## Mode degrade (BM25 pur)

Sans provider d'embeddings disponible :

- la recherche vectorielle (`vector`) et la fusion (`hybrid`) retombent
  automatiquement sur BM25 ;
- `HybridFuser` avec `mode="bm25"` retourne simplement les resultats BM25.

## Fusion hybride (Section 7.3)

Deux strategies, pilotees par `hybrid_method` (config) ou le paramètre `mode`
de `fuse_hybrid` :

- **RRF** (`mode="rrf"`) : `score = sum(1 / (k + rank_i))`, k=60 par defaut.
   Ideal quand BM25 et les vecteurs ont des echelles de score incomparables.
- **Ponderee** (`mode="weighted"`) : `score = alpha * bm25_norm + (1 - alpha) * vector_norm`,
  avec normalisation min-max. `alpha = hybrid_weight_bm25` par defaut (0.65).
