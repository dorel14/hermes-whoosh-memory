"""Indexation et recherche d'images pour whoosh-memory (Section spec images v0.2).

Permet à l'agent de retrouver des images/photos par recherche textuelle ou
sémantique, puis de les proposer ou de les analyser via le MCP vision existant
(qui attend du base64).

Principes :
- Stockage dans l'index : chemin absolu + métadonnées + embedding (jamais le
  base64 ni le binaire).
- Outil vision : MCP vision existant — le plugin fournit le base64 à la volée.
- Recherche : hybride possible (texte OCR / description + embedding visuel).
- Localité : tout reste local. La conversion base64 se fait uniquement au
  moment où l'agent en a besoin.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from .config import DEFAULTS
from .extractors import chunk_text
from .schema import DEFAULT_EMBEDDING_DIM
from .search import _Index, SearchHit, whoosh_search

logger = logging.getLogger("hermes.plugins.whoosh-memory.images")

# Extensions d'images supportées.
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif",
}

# Taille max pour l'encodage base64 retourné par whoosh_image_base64.
DEFAULT_BASE64_MAX_MB = 10
DEFAULT_MAX_IMAGE_MB = 20
DEFAULT_MAX_DIMENSION = 1024


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _content_hash_for_image(path: Path) -> str:
    """ID stable = hash du chemin + mtime + taille."""
    stat = path.stat()
    raw = f"{str(path.resolve())}|{stat.st_mtime_ns}|{stat.st_size}".encode()
    return "img_" + hashlib.sha256(raw).hexdigest()[:24]


def image_metadata(path: Path) -> dict:
    """Métadonnées de l'image (taille, dimensions, format, etc.)."""
    try:
        with Image.open(path) as im:
            return {
                "width": im.width,
                "height": im.height,
                "format": im.format,
                "mode": im.mode,
                "size_bytes": path.stat().st_size,
            }
    except Exception:
        return {"size_bytes": path.stat().st_size}


def image_to_base64(
    path: Path, max_size_mb: float = DEFAULT_BASE64_MAX_MB
) -> tuple[str, dict]:
    """Lit une image et retourne (base64, métadonnées).

    Redimensionne si nécessaire pour rester sous max_size_mb. Ne stocke jamais
    le base64 dans l'index ni dans les logs.
    """
    meta = image_metadata(path)
    with Image.open(path) as im:
        # Convertir en RGB pour un JPEG compatible si nécessaire.
        if im.mode in ("RGBA", "P", "LA"):
            im = im.convert("RGB")

        # Redimensionner si le fichier est trop volumineux ou trop grand.
        max_dim = DEFAULT_MAX_DIMENSION
        if max(im.width, im.height) > max_dim:
            im.thumbnail((max_dim, max_dim))

        # Qualité progressive pour respecter la limite de taille.
        quality = 90
        while True:
            buf = __import__("io").BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            size_mb = len(buf.getvalue()) / (1024 * 1024)
            if size_mb <= max_size_mb or quality <= 40:
                break
            quality -= 10

        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    meta["base64_bytes"] = len(buf.getvalue())
    meta["base64_chars"] = len(b64)
    return b64, meta


def index_images(
    index_dir: str | os.PathLike[str],
    image_paths: Iterable[str | os.PathLike[str]],
    *,
    embeddings=None,
    max_image_mb: float = DEFAULT_MAX_IMAGE_MB,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    ocr: bool = False,
    dry_run: bool = False,
) -> dict:
    """Indexe une liste de chemins d'images.

    Retourne un rapport : indexés, ignorés, échoués, dry_run_total.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "indexed": 0,
        "skipped": 0,
        "errors": [],
        "dry_run_total": 0,
    }

    # Charger l'index existant pour éviter les doubles indexations.
    index = _Index(index_dir, dimension=DEFAULT_EMBEDDING_DIM)

    for p in image_paths:
        p = Path(p)
        report["dry_run_total"] += 1

        if not p.exists() or not p.is_file():
            report["skipped"] += 1
            report["errors"].append({"path": str(p), "reason": "not_found"})
            continue

        if not _is_image_file(p):
            report["skipped"] += 1
            continue

        # Vérification taille max.
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > max_image_mb:
                report["skipped"] += 1
                report["errors"].append(
                    {"path": str(p), "reason": f"too_large:{size_mb:.1f}MB"}
                )
                continue
        except OSError as exc:
            report["skipped"] += 1
            report["errors"].append({"path": str(p), "reason": str(exc)})
            continue

        doc_id = _content_hash_for_image(p)
        meta = image_metadata(p)

        if dry_run:
            report["indexed"] += 1
            continue

        # OCR optionnel (désactivé par défaut — pas de dépendance Tesseract).
        content = ""
        if ocr:
            try:
                content = _ocr_image(p)
            except Exception as exc:  # pragma: no cover - OCR optionnel
                logger.debug("OCR échoué pour %s: %s", p, exc)

        # Embedding visuel optionnel (délégé à libembedding-ng / fastembed).
        embedding = None
        if embeddings is not None and getattr(embeddings, "available", False):
            try:
                embedding = embeddings.embed_image(str(p))
            except Exception:
                # embed_image peut ne pas exister selon le provider.
                try:
                    embedding = embeddings.embed(f"image:{p}")
                except Exception:
                    embedding = None

        index.add(
            doc_id=doc_id,
            path=str(p.resolve()),
            title=p.name,
            text=content,
            embedding=embedding,
        )
        report["indexed"] += 1

    index.finish()
    return report


def _ocr_image(path: Path) -> str:
    """OCR optionnel via Tesseract (si disponible). Retourne le texte."""
    import subprocess

    result = subprocess.run(
        ["tesseract", str(path), "stdout", "-l", "fra+eng"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def whoosh_image_info(
    index_dir: str | os.PathLike[str], path_or_id: str
) -> Optional[dict]:
    """Métadonnées d'une image indexée (chemin ou id)."""
    index_dir = Path(index_dir)
    index = _Index(index_dir, dimension=DEFAULT_EMBEDDING_DIM)
    index.finish()

    # Recherche par chemin ou par id dans le store BM25 pur.
    if index._py is not None:
        for doc in index._py.docs:
            if doc.get("id") == path_or_id or doc.get("path") == path_or_id:
                return {
                    "id": doc.get("id"),
                    "path": doc.get("path"),
                    "title": doc.get("title"),
                    "has_embedding": bool(doc.get("embedding")),
                }
    return None


def whoosh_image_base64(
    path_or_id: str,
    *,
    max_size_mb: float = DEFAULT_BASE64_MAX_MB,
) -> Optional[dict]:
    """Convertit une image (par chemin absolu) en base64 pour le MCP vision.

    Vérifie que le fichier existe et est une image supportée avant lecture.
    Ne stocke jamais le base64.
    """
    p = Path(path_or_id)
    if not p.exists() or not p.is_file():
        return None
    if not _is_image_file(p):
        return None

    try:
        b64, meta = image_to_base64(p, max_size_mb=max_size_mb)
    except Exception as exc:
        logger.error("Échec conversion base64 pour %s: %s", p, exc)
        return None

    return {
        "path": str(p.resolve()),
        "title": p.name,
        "base64": b64,
        "mime": "image/jpeg",
        "metadata": meta,
    }


def search_images(
    index_dir: str | os.PathLike[str],
    query: str,
    *,
    top_k: int = 10,
    hybrid: str | bool = "auto",
    embeddings=None,
) -> list[SearchHit]:
    """Recherche d'images par texte (filtre type=image via le titre/chemin)."""
    return whoosh_search(
        index_dir,
        query,
        top_k=top_k,
        hybrid=hybrid,
        embeddings=embeddings,
    )

__all__ = [
    "IMAGE_EXTENSIONS",
    "image_metadata",
    "image_to_base64",
    "index_images",
    "whoosh_image_info",
    "whoosh_image_base64",
    "search_images",
]
