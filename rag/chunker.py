"""Split documents into overlapping, paragraph-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    index: int  # position of the chunk inside its source document


def _split_long(text: str, size: int) -> list[str]:
    """Split a paragraph that is longer than `size` on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?。！？])\s*", text)
    parts, buf = [], ""
    for sent in sentences:
        if not sent:
            continue
        if len(buf) + len(sent) + 1 > size and buf:
            parts.append(buf.strip())
            buf = ""
        while len(sent) > size:  # hard split for very long sentences
            parts.append(sent[:size])
            sent = sent[size:]
        buf = f"{buf} {sent}" if buf else sent
    if buf.strip():
        parts.append(buf.strip())
    return parts


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Group paragraphs into chunks of roughly `chunk_size` characters.

    Consecutive chunks share up to `overlap` characters of trailing context so
    that answers spanning a boundary can still be retrieved.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(p, chunk_size) if len(p) > chunk_size else [p])

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > chunk_size:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            # start the overlap on a word boundary when possible
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            buf = f"{tail}\n\n{piece}" if tail else piece
        else:
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        chunks.append(buf)

    return [Chunk(id=f"{source}#{i}", source=source, text=c, index=i) for i, c in enumerate(chunks)]
