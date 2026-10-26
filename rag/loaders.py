"""Load plain-text content from .txt, .md and .pdf files."""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


class UnsupportedFileType(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Return the text content of a document given its name and raw bytes."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)
    return data.decode("utf-8", errors="replace")


def load_file(path: str | Path) -> str:
    path = Path(path)
    return extract_text(path.name, path.read_bytes())


def iter_documents(folder: str | Path):
    """Yield (filename, text) for every supported file in a folder (recursive)."""
    folder = Path(folder)
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path.name, load_file(path)
