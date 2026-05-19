"""Small in-memory lexical store for local RAG.

This is intentionally dependency-free. It is not a true embedding database, but
it gives the project a safe retrieval layer with symbol metadata filtering.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from rag.document_loader import Document


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class SearchResult:
    """A retrieved document chunk."""

    document: Document
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["document"] = self.document.to_dict()
        return data


class LocalVectorStore:
    """A lightweight TF-IDF-like retriever over loaded documents."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._vectors = [_term_counts(document.text) for document in documents]
        self._idf = self._build_idf(self._vectors)

    def search(
        self,
        query: str,
        symbol: str,
        top_k: int = 5,
        include_general: bool = True,
    ) -> list[SearchResult]:
        """Search documents with strict company-symbol filtering."""

        clean_symbol = symbol.strip().upper()
        query_vector = _term_counts(query)
        if not query_vector:
            return []

        results: list[SearchResult] = []
        for document, vector in zip(self.documents, self._vectors):
            if not _document_allowed(document, clean_symbol, include_general):
                continue
            score = _cosine_similarity(query_vector, vector, self._idf)
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    document=document,
                    score=score,
                    snippet=_make_snippet(document.text, query_vector),
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _build_idf(vectors: list[Counter[str]]) -> dict[str, float]:
        document_count = len(vectors)
        document_frequency: Counter[str] = Counter()
        for vector in vectors:
            document_frequency.update(vector.keys())

        return {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }


def _document_allowed(document: Document, symbol: str, include_general: bool) -> bool:
    if document.scope == "general":
        return include_general
    return document.scope == "company" and document.symbol == symbol


def _term_counts(text: str) -> Counter[str]:
    tokens = [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 1
    ]
    return Counter(tokens)


def _cosine_similarity(
    query_vector: Counter[str],
    document_vector: Counter[str],
    idf: dict[str, float],
) -> float:
    shared_terms = set(query_vector) & set(document_vector)
    if not shared_terms:
        return 0.0

    numerator = sum(
        query_vector[term] * document_vector[term] * idf.get(term, 1.0) ** 2
        for term in shared_terms
    )
    query_norm = math.sqrt(
        sum((count * idf.get(term, 1.0)) ** 2 for term, count in query_vector.items())
    )
    document_norm = math.sqrt(
        sum((count * idf.get(term, 1.0)) ** 2 for term, count in document_vector.items())
    )

    if query_norm == 0 or document_norm == 0:
        return 0.0
    return numerator / (query_norm * document_norm)


def _make_snippet(text: str, query_vector: Counter[str], max_chars: int = 500) -> str:
    lowered = text.lower()
    first_match = min(
        (lowered.find(term) for term in query_vector if lowered.find(term) >= 0),
        default=0,
    )
    start = max(first_match - 120, 0)
    snippet = text[start : start + max_chars].strip()
    return " ".join(snippet.split())
