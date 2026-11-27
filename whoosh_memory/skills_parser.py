"""Parser Markdown minimal (Section 4.4) — fichiers .md de skill.

Extrait le titre (H1, sinon première ligne, sinon nom de fichier),
les headings et le contenu texte brut d'un fichier de skill.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .extractors import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    _read_text,
    chunk_text,
)

_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n", re.S)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.M)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class ParsedSkill:
    path: str
    title: str
    headings: list[tuple[int, str]] = field(default_factory=list)
    content: str = ""
    chunks: list[str] = field(default_factory=list)


def parse_skill_md(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ParsedSkill:
    """Parse un fichier .md de skill : titre, headings, contenu texte."""
    p = Path(path)
    raw = _read_text(p)

    # enlève le frontmatter YAML s'il est en tête de fichier
    body = _FRONTMATTER_RE.sub("", raw, count=1).strip()

    # headings : niveau et titre
    headings: list[tuple[int, str]] = []
    for match in _HEADING_RE.finditer(body):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        headings.append((level, heading_text))

    # titre : premier H1, sinon première ligne non vide, sinon nom du fichier
    title = ""
    for level, heading_text in headings:
        if level == 1:
            title = heading_text
            break
    if not title:
        for line in body.splitlines():
            line = line.strip()
            if line:
                title = line[:120]
                break
    if not title:
        title = p.stem

    # contenu texte : code blocs (contenu conservé, fences retirées),
    # balises HTML enlevées, blancs redressés
    text = _CODE_FENCE_RE.sub(lambda m: m.group(0).strip("`").strip() + "\n", body)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    content = text.strip()

    parsed = ParsedSkill(
        path=str(p),
        title=title,
        headings=headings,
        content=content,
    )
    parsed.chunks = chunk_text(content, size=chunk_size, overlap=chunk_overlap)
    return parsed
