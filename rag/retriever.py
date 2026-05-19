"""High-level retrieval functions for researcher agents."""

from __future__ import annotations

from typing import Any

from rag.document_loader import DEFAULT_DOCUMENT_DIR, load_documents
from rag.vector_store import LocalVectorStore, SearchResult


DEFAULT_TOP_K = 5


def retrieve_context(
    symbol: str,
    query: str | None = None,
    document_dir: str = DEFAULT_DOCUMENT_DIR,
    top_k: int = DEFAULT_TOP_K,
    include_general: bool = True,
) -> dict[str, Any]:
    """Retrieve safe local context for a ticker.

    Company documents are only returned when their metadata symbol matches the
    requested ticker. General documents can be shared across symbols.
    """

    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("symbol cannot be empty")

    documents = load_documents(document_dir)
    if not documents:
        return {
            "symbol": clean_symbol,
            "document_dir": document_dir,
            "results": [],
            "text": "No local knowledge-base documents found.",
            "data_gap": f"No local knowledge-base documents found in {document_dir}.",
        }

    search_query = query or _default_query(clean_symbol)
    store = LocalVectorStore(documents)
    results = store.search(
        query=search_query,
        symbol=clean_symbol,
        top_k=top_k,
        include_general=include_general,
    )

    company_doc_count = sum(
        1 for document in documents if document.scope == "company" and document.symbol == clean_symbol
    )
    general_doc_count = sum(1 for document in documents if document.scope == "general")
    data_gap = None
    if company_doc_count == 0:
        data_gap = (
            f"No {clean_symbol}-specific documents found in the local knowledge base. "
            "Only general documents may be used."
        )

    return {
        "symbol": clean_symbol,
        "document_dir": document_dir,
        "query": search_query,
        "company_documents": company_doc_count,
        "general_documents": general_doc_count,
        "results": [result.to_dict() for result in results],
        "text": format_retrieval_context(results, data_gap=data_gap),
        "data_gap": data_gap,
    }


def _default_query(symbol: str) -> str:
    return (
        f"{symbol} company filing revenue margin cash flow balance sheet risk "
        "competition management outlook valuation investment thesis"
    )


def format_retrieval_context(
    results: list[SearchResult],
    data_gap: str | None = None,
) -> str:
    """Format retrieval results for an agent prompt."""

    lines: list[str] = []
    if data_gap:
        lines.append(data_gap)

    if not results:
        lines.append("No relevant local documents retrieved.")
        return "\n".join(lines)

    lines.append("Retrieved local knowledge-base context:")
    for index, result in enumerate(results, start=1):
        document = result.document
        symbol_text = f", symbol={document.symbol}" if document.symbol else ""
        lines.append(
            f"{index}. {document.title} "
            f"(scope={document.scope}{symbol_text}, type={document.doc_type or 'unknown'}, "
            f"score={result.score:.3f}, path={document.path})\n"
            f"   {result.snippet}"
        )

    return "\n".join(lines)
