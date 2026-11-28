"""Extraction texte multi-format (Section 4.3) — plugin whoosh-memory.

Formats supportés : PDF, DOCX, Markdown, code source (texte brut).
Contraintes : denylist de chemins (identique en esprit au plugin recall),
taille max par fichier (max_file_mb), nombre max de fichiers (max_files),
budget temps (budget_seconds). Découpage commun ~1000 caractères, overlap 150.
"""

from __future__ import annotations

import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_MAX_FILE_MB = 10.0
DEFAULT_MAX_FILES = 1000
DEFAULT_BUDGET_SECONDS = 300.0
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Denylist identique en esprit au plugin recall : répertoires système,
# caches, gestionnaires de paquets, dépôts et environnements virtuels.
DEFAULT_DENY_PATTERNS: list[str] = [
    r"(^|/)\.git($|/)",
    r"(^|/)\.hg($|/)",
    r"(^|/)\.svn($|/)",
    r"(^|/)\.bzr($|/)",
    r"(^|/)\.cache($|/)",
    r"(^|/)\.npm($|/)",
    r"(^|/)\.yarn($|/)",
    r"(^|/)\.pnpm($|/)",
    r"(^|/)node_modules($|/)",
    r"(^|/)__pycache__($|/)",
    r"(^|/)\.venv($|/)",
    r"(^|/)venv($|/)",
    r"(^|/)\.env($|/)",
    r"(^|/)\.tox($|/)",
    r"(^|/)\.mypy_cache($|/)",
    r"(^|/)\.pytest_cache($|/)",
    r"(^|/)\.ruff_cache($|/)",
    r"(^|/)\.idea($|/)",
    r"(^|/)\.vscode($|/)",
    r"(^|/)\.DS_Store$",
    r"(^|/)proc($|/)",
    r"(^|/)sys($|/)",
    r"(^|/)dev($|/)",
    r"(^|/)tmp($|/)",
    r"(^|/)(var|etc|run|root)(/|$)",
]

# Extensions toujours ignorées (binaires, archives, médias).
DENY_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin", ".o", ".a", ".obj",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp", ".avif",
    ".mp3", ".mp4", ".wav", ".flac", ".avi", ".mov", ".mkv", ".wmv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".ear",
    ".iso", ".img",
}

# Extensions que l'extracteur sait traiter.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".md", ".markdown",
    ".txt", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".rb", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".sql", ".rst", ".org",
}


def compile_deny_patterns(patterns: Optional[list[str]] = None) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in (patterns or DEFAULT_DENY_PATTERNS)]


def is_denied(path: str | Path, patterns: Optional[list[re.Pattern[str]]] = None) -> bool:
    """Vérifie si un chemin correspond au denylist (identique plugin recall)."""
    pats = patterns or compile_deny_patterns()
    p = str(Path(path))
    if any(pat.search(p) for pat in pats):
        return True
    if Path(p).suffix.lower() in DENY_EXTENSIONS:
        return True
    return False


@dataclass
class ExtractedDocument:
    path: str
    title: str
    mime: str
    size_bytes: int
    text: str
    error: Optional[str] = None
    skipped: bool = False
    chunks: list[str] = field(default_factory=list)


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_pdf(path: Path) -> str:
    parts: list[str] = []
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(str(path))
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        pass
    try:
        import fitz  # type: ignore[import-not-found]

        doc = fitz.open(str(path))
        for page in doc:
            parts.append(page.get_text())
        return "\n".join(parts)
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore[import-not-found]

        return extract_text(str(path))
    except Exception:
        pass
    return ""


def extract_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore[import-not-found]

        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    except Exception:
        return _read_text(path)


def extract_markdown(path: Path) -> str:
    return _read_text(path)


def extract_code(path: Path) -> str:
    return _read_text(path)


def extract_file(path: str | Path) -> ExtractedDocument:
    p = Path(path)
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or ""
    text = ""
    error: Optional[str] = None
    try:
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            text = extract_pdf(p)
        elif suffix == ".docx":
            text = extract_docx(p)
        elif suffix in (".md", ".markdown"):
            text = extract_markdown(p)
        else:
            text = extract_code(p)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    try:
        size_bytes = p.stat().st_size
    except OSError:
        size_bytes = 0
    return ExtractedDocument(
        path=str(p),
        title=p.name,
        mime=mime,
        size_bytes=size_bytes,
        text=text,
        error=error,
        skipped=bool(error),
    )


def iter_extractable_files(
    root: str | Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    deny_patterns: Optional[list[re.Pattern[str]]] = None,
    allowed_extensions: Optional[set[str]] = None,
) -> Iterator[Path]:
    root = Path(root)
    pats = deny_patterns or compile_deny_patterns()
    allowed = allowed_extensions or SUPPORTED_EXTENSIONS
    count = 0
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _: None):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and not is_denied(Path(dirpath) / d, patterns=pats)
        ]
        for name in sorted(filenames):
            if count >= max_files:
                return
            full = Path(dirpath) / name
            if name.startswith(".") or full.is_symlink():
                continue
            if is_denied(full, patterns=pats):
                continue
            if full.suffix.lower() not in allowed:
                continue
            try:
                if not full.is_file():
                    continue
            except OSError:
                continue
            count += 1
            yield full


def extract_directory(
    root: str | Path,
    *,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
    max_files: int = DEFAULT_MAX_FILES,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    deny_patterns: Optional[list[re.Pattern[str]]] = None,
) -> list[ExtractedDocument]:
    root = Path(root)
    start = time.monotonic()
    docs: list[ExtractedDocument] = []
    pats = deny_patterns or compile_deny_patterns()
    for path in iter_extractable_files(
        root, max_files=max_files, deny_patterns=pats
    ):
        if time.monotonic() - start > budget_seconds:
            break
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        if size_bytes / (1024 * 1024) > max_file_mb:
            docs.append(
                ExtractedDocument(
                    path=str(path),
                    title=path.name,
                    mime="",
                    size_bytes=size_bytes,
                    text="",
                    error=f"file too large ({size_bytes / (1024 * 1024):.1f} MB > {max_file_mb} MB)",
                    skipped=True,
                )
            )
            continue
        docs.append(extract_file(path))
    return docs


def chunk_text(
    text: str,
    *,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Découpe un texte en morceaux de ~size caractères avec overlap."""
    if not text:
        return []
    if overlap >= size:
        overlap = max(size // 10, 1)
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        if end < n:
            cut = max(
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind("! "),
                chunk.rfind("? "),
            )
            if cut > size // 2:
                end = start + cut + 1
                chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks