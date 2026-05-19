"""Load local documents for retrieval-augmented research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DOCUMENT_DIR = "data/documents"
SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    """A local knowledge-base document with routing metadata."""

    path: str
    title: str
    text: str
    scope: str
    symbol: str | None = None
    doc_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_documents(document_dir: str = DEFAULT_DOCUMENT_DIR) -> list[Document]:
    """Load supported text documents from the knowledge base directory."""

    root = Path(document_dir)
    if not root.exists():
        return []

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        metadata = _metadata_from_text(text)
        filename_metadata = _metadata_from_filename(path)
        metadata = {**filename_metadata, **metadata}

        scope = str(metadata.get("scope") or "general").lower()
        symbol = metadata.get("symbol")
        if symbol:
            symbol = str(symbol).upper()
            scope = "company"

        documents.append(
            Document(
                path=str(path),
                title=str(metadata.get("title") or path.stem),
                text=_strip_front_matter(text),
                scope=scope,
                symbol=symbol,
                doc_type=metadata.get("doc_type"),
            )
        )

    return documents


def _metadata_from_text(text: str) -> dict[str, str]:
    """Parse simple YAML-like front matter from a document."""

    if not text.startswith("---"):
        return {}

    lines = text.splitlines()
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip().strip('"').strip("'")
    return metadata


def _metadata_from_filename(path: Path) -> dict[str, str]:
    """Infer metadata from names like AAPL_10K_2025.md or general_framework.md."""

    stem = path.stem
    parts = [part for part in stem.replace("-", "_").split("_") if part]
    if not parts:
        return {}

    first = parts[0].upper()
    if first == "GENERAL":
        return {"scope": "general", "doc_type": "_".join(parts[1:]) if len(parts) > 1 else "general"}

    if first.isalpha() and 1 <= len(first) <= 6:
        return {
            "scope": "company",
            "symbol": first,
            "doc_type": "_".join(parts[1:]) if len(parts) > 1 else "company_note",
        }

    return {"scope": "general"}


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text

    lines = text.splitlines()
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :]).strip()
    return text
