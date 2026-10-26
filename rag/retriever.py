"""Retrievers. BM25 (pure Python, default) and an optional embedding retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunker import Chunk

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "our that the this to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens, plus character bigrams for CJK text."""
    text = text.lower()
    tokens = [t for t in _WORD.findall(text) if t not in STOPWORDS]
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class BM25Retriever:
    """Okapi BM25 over in-memory chunks. No external dependencies."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tfs: list[Counter] = []
        self._lens: list[int] = []
        self._df: Counter = Counter()

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tf = Counter(tokenize(chunk.text))
            self.chunks.append(chunk)
            self._tfs.append(tf)
            self._lens.append(sum(tf.values()))
            self._df.update(tf.keys())

    def remove_source(self, source: str) -> int:
        keep = [c for c in self.chunks if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.__init__(self.k1, self.b)
            self.add(keep)
        return removed

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        n = len(self.chunks)
        avgdl = sum(self._lens) / n or 1.0
        q_terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self._tfs):
            score = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * self._lens[i] / avgdl)
                score += idf * f * (self.k1 + 1) / denom
            if score > 0:
                scores.append(SearchResult(self.chunks[i], score))
        scores.sort(key=lambda r: r.score, reverse=True)
        return scores[:top_k]


class EmbeddingRetriever:
    """Dense retrieval with sentence-transformers (optional, runs locally).

    Install with: pip install sentence-transformers
    """

    name = "embedding"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy, optional

        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self._vecs = None

    def add(self, chunks: list[Chunk]) -> None:
        import numpy as np

        if not chunks:
            return
        vecs = self.model.encode([c.text for c in chunks], normalize_embeddings=True)
        self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])
        self.chunks.extend(chunks)

    def remove_source(self, source: str) -> int:
        keep_idx = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep_idx)
        if removed:
            self.chunks = [self.chunks[i] for i in keep_idx]
            self._vecs = self._vecs[keep_idx] if keep_idx else None
        return removed

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self._vecs @ q
        order = sims.argsort()[::-1][:top_k]
        return [SearchResult(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0]


def make_retriever(kind: str = "bm25"):
    if kind == "embedding":
        return EmbeddingRetriever()
    return BM25Retriever()
