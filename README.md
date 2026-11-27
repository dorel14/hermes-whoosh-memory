# whoosh-memory — Mémoire locale hybride pour Hermes Agent

Un plugin Hermes 100 % local qui indexe l'historique des sessions, MEMORY.md / USER.md,
les documents locaux, les skills Hermes et les images dans un index whoosh-ng
(BM25 + recherche vectorielle native) pour une récupération hors-ligne rapide
avec prefetch automatique avant chaque turn.

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

### Déploiement pour Lina (sur l'hôte Hermes)

```bash
cd /opt/data/plugins/hermes-whoosh-memory
pip install -e .
# puis déclarer le plugin dans la config de Lina
```

## Utilisation CLI

```bash
whoosh-memory init              # créer les dossiers + config par défaut
whoosh-memory config            # afficher la configuration
whoosh-memory search "requête"  # rechercher dans l'index
whoosh-memory index ./docs      # indexer un dossier de documents
whoosh-memory index-images ./Photos  # indexer un dossier d'images
whoosh-memory stats             # statistiques de l'index
```

## Outils agents (MCP / registre Hermes)

| Outil | Rôle |
|-------|------|
| `whoosh_search` | Recherche hybride (texte + vecteur) dans l'index |
| `whoosh_open` | Lire un document par id ou path |
| `whoosh_forget` | Supprimer un document de l'index |
| `whoosh_stats` | Statistiques de l'index |
| `whoosh_add` | Ajouter un document manuellement |
| `whoosh_image_base64` | Convertir une image indexée en base64 pour le MCP vision |
| `whoosh_image_info` | Métadonnées d'une image indexée |

## Indexation d'images (spec v0.2)

Les images sont indexées avec chemin absolu + métadonnées + embedding optionnel
(jamais le base64 ni le binaire stocké). La conversion base64 se fait à la
volée via `whoosh_image_base64` pour le MCP vision existant.

Configuration (dans `whoosh-memory.json` sous `HERMES_HOME/plugin-data/whoosh-memory/`) :

```json
{
  "index_images": true,
  "index_image_paths": ["~/Photos", "~/Pictures", "~/screenshots"],
  "image_ocr": false,
  "image_embedding": true,
  "image_embedding_provider": "libembedding-ng",
  "image_max_mb": 20,
  "image_max_dimension": 1024,
  "image_base64_max_mb": 10
}
```

## Sécurité

Les chemins sous `HERMES_HOME` contenant des credentials sont exclus par une
denylist stricte. Aucun identifiant, clé d'API ou jeton MCP n'est stocké en
clair : ils sont anonymisés par `[REDACTED]`.

## Dépendances

- Requis : `whoosh-ng>=0.3.0`
- Optionnel : `fastembed`, `sentence-transformers`, `ollama`, `Pillow`

```bash
pip install "hermes-whoosh-memory[all]"
```

## Licence

MIT — voir `LICENSE`.
