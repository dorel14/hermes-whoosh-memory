"""Interface en ligne de commande pour whoosh-memory.

Usage :
    whoosh-memory init                # créer le dossier de config par défaut
    whoosh-memory config              # afficher la config courante
    whoosh-memory search <requête>    # rechercher dans l'index
    whoosh-memory index <dossier>     # indexer un dossier de documents
    whoosh-memory index-images <dossier>  # indexer un dossier d'images
    whoosh-memory stats               # statistiques de l'index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CONFIG_PATH, DEFAULTS, hermes_home, load_config
from .search import whoosh_search


def _default_index_dir() -> Path:
    return hermes_home() / "plugin-data" / "whoosh-memory" / "index"


def cmd_init(args: argparse.Namespace) -> int:
    pdata = hermes_home() / "plugin-data" / "whoosh-memory"
    (pdata / "index").mkdir(parents=True, exist_ok=True)
    (pdata / "skills-index").mkdir(parents=True, exist_ok=True)
    (pdata / "embeddings").mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(DEFAULTS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Config créée : {CONFIG_PATH}")
    print(f"Index : {_default_index_dir()}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config()
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index_dir = args.index or _default_index_dir()
    hits = whoosh_search(
        str(index_dir),
        args.query,
        top_k=args.limit,
        hybrid=args.hybrid,
    )
    if not hits:
        print("Aucun résultat.")
        return 0
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.title} ({h.source}) score={h.score}")
        print(f"    id={h.doc_id}  path={h.path}")
        if h.snippet:
            print(f"    > {h.snippet[:200]}")
        print()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from .extractors import extract_directory
    from .search import index_documents

    index_dir = args.index or _default_index_dir()
    docs = extract_directory(args.path)
    count = index_documents(str(index_dir), docs)
    print(f"{count} chunks indexés depuis {args.path}")
    return 0


def cmd_index_images(args: argparse.Namespace) -> int:
    from .images import index_images

    index_dir = args.index or _default_index_dir()
    paths = [str(p) for p in Path(args.path).rglob("*") if p.is_file()]
    report = index_images(str(index_dir), paths, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    index_dir = args.index or _default_index_dir()
    from .search import _Index

    idx = _Index(Path(index_dir))
    idx.finish()
    n = len(idx._py.docs) if idx._py is not None else "?"
    print(f"Documents indexés : {n}")
    print(f"Index dir : {index_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whoosh-memory")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--index", default=None, help="répertoire de l'index")

    p_init = sub.add_parser("init", help="initialiser les dossiers et config")
    p_init.set_defaults(func=cmd_init)

    p_cfg = sub.add_parser("config", help="afficher la configuration")
    p_cfg.set_defaults(func=cmd_config)

    p_search = sub.add_parser("search", help="rechercher dans l'index")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--hybrid", default="auto")
    add_common(p_search)
    p_search.set_defaults(func=cmd_search)

    p_index = sub.add_parser("index", help="indexer un dossier de documents")
    p_index.add_argument("path")
    add_common(p_index)
    p_index.set_defaults(func=cmd_index)

    p_img = sub.add_parser("index-images", help="indexer un dossier d'images")
    p_img.add_argument("path")
    p_img.add_argument("--dry-run", action="store_true")
    add_common(p_img)
    p_img.set_defaults(func=cmd_index_images)

    p_stats = sub.add_parser("stats", help="statistiques de l'index")
    add_common(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
