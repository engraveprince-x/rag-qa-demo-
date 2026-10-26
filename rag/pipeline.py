"""Glue: ingest documents -> chunk -> index -> retrieve -> (optionally) generate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_text
from .llm import LLMClient, LLMConfig
from .loaders import extract_text, iter_documents
import re

from .retriever import SearchResult, make_retriever, tokenize

log = logging.getLogger(__name__)


@dataclass
class Source:
    n: int
    source: str
    chunk_id: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    mode: str  # "llm" | "retrieval-only" | "no-results"
    sources: list[Source] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "note": self.note,
            "sources": [s.__dict__ for s in self.sources],
        }


class RAGPipeline:
    def __init__(
        self,
        retriever_kind: str = "bm25",
        llm_config: LLMConfig | None = None,
        chunk_size: int = 800,
        overlap: int = 150,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = make_retriever(retriever_kind)
        self.llm_config = llm_config or LLMConfig.from_env()
        self.llm = llm_client or (LLMClient(self.llm_config) if self.llm_config.enabled else None)
        self.chunk_size, self.overlap = chunk_size, overlap
        self.documents: dict[str, int] = {}  # source -> number of chunks

    # ---------- ingestion ----------
    def add_text(self, source: str, text: str) -> int:
        if source in self.documents:
            self.retriever.remove_source(source)
        chunks = chunk_text(text, source, self.chunk_size, self.overlap)
        self.retriever.add(chunks)
        self.documents[source] = len(chunks)
        return len(chunks)

    def add_file_bytes(self, filename: str, data: bytes) -> int:
        return self.add_text(filename, extract_text(filename, data))

    def add_folder(self, folder: str | Path) -> int:
        return sum(self.add_text(name, text) for name, text in iter_documents(folder))

    def remove(self, source: str) -> bool:
        if source not in self.documents:
            return False
        self.retriever.remove_source(source)
        del self.documents[source]
        return True

    # ---------- querying ----------
    def ask(self, question: str, top_k: int = 4) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        results: list[SearchResult] = self.retriever.search(question, top_k=top_k)
        sources = [
            Source(n=i + 1, source=r.chunk.source, chunk_id=r.chunk.id, score=round(r.score, 4), text=r.chunk.text)
            for i, r in enumerate(results)
        ]
        if not sources:
            return Answer(question, "No relevant passages were found in the knowledge base.", "no-results")

        if self.llm is not None:
            context = "\n\n".join(f"[{s.n}] (source: {s.source})\n{s.text}" for s in sources)
            try:
                return Answer(question, self.llm.complete(question, context), "llm", sources)
            except Exception as exc:  # network / auth errors -> graceful fallback
                log.warning("LLM call failed, falling back to retrieval-only: %s", exc)
                note = f"LLM call failed ({type(exc).__name__}); showing retrieved passages instead."
        else:
            note = "No LLM API key configured; showing the most relevant passages with their sources."

        lines = [f"[{s.n}] {s.source}: {best_sentences(s.text, question)}" for s in sources]
        return Answer(question, "\n\n".join(lines), "retrieval-only", sources, note)


def best_sentences(text: str, question: str, max_sentences: int = 2, limit: int = 400) -> str:
    """Extractive fallback: return the sentences of `text` that best match the question."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if s.strip()]
    sentences = [s for s in sentences if not s.startswith("#")] or sentences
    q = set(tokenize(question))
    scored = [(len(q & set(tokenize(s))), i, s) for i, s in enumerate(sentences)]
    if any(sc for sc, _, _ in scored):
        scored = [x for x in scored if x[0] > 0]
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_sentences]
    picked = " ".join(s for _, _, s in sorted(top, key=lambda x: x[1]))
    picked = " ".join(picked.split())
    return picked if len(picked) <= limit else picked[:limit].rsplit(" ", 1)[0] + " …"
