from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.documents import Document

# Bloco considerado "cabeçalho/rodapé repetido" se aparecer em mais
# do que esta fração do total de páginas do documento.
_REPEATED_BLOCK_THRESHOLD = 0.10  # 10 %
_MIN_TEXT_LENGTH = 5  # caracteres mínimos para considerar um bloco


def _normalize(text: str) -> str:
    """Remove espaços extra e converte para minúsculas para comparação."""
    return " ".join(text.lower().split())


def _find_repeated_texts(
    blocks: list[dict[str, Any]],
    total_pages: int,
) -> set[str]:
    """
    Identifica textos que se repetem em demasiadas páginas
    (cabeçalhos, rodapés, numeração de página, etc.).
    """
    text_page_count: Counter[str] = Counter()
    seen: set[tuple[str, int]] = set()

    for block in blocks:
        text = _normalize(block.get("text", ""))
        page = block.get("page_number", 0)
        key = (text, page)
        if key not in seen:
            seen.add(key)
            text_page_count[text] += 1

    threshold = max(2, int(total_pages * _REPEATED_BLOCK_THRESHOLD))
    return {text for text, count in text_page_count.items() if count >= threshold}


def process_blocks_to_langchain_docs(
    blocks: list[dict[str, Any]],
    source_name: str = "documento",
) -> list[Document]:
    """
    Convert OCR blocks into LangChain documents with traceable metadata.

    Filters out:
    - Empty or very short blocks
    - Repeated headers/footers (appear on >10% of pages)
    """
    if not blocks:
        return []

    total_pages = max((b.get("page_number", 0) or 0 for b in blocks), default=1)
    repeated = _find_repeated_texts(blocks, total_pages)

    documents: list[Document] = []
    skipped_repeated = 0
    skipped_short = 0

    for i, block in enumerate(blocks):
        text = block.get("text", "").strip()

        # Filtro 1: blocos vazios ou muito curtos
        if len(text) < _MIN_TEXT_LENGTH:
            skipped_short += 1
            continue

        # Filtro 2: cabeçalhos/rodapés repetidos
        if _normalize(text) in repeated:
            skipped_repeated += 1
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source": source_name,
                "block_id": i,
                "page_number": block.get("page_number"),
                "coords": block.get("coords"),
                "confidence": block.get("confidence"),
                "type": "text_block",
                "extraction_method": "ocr",
            },
        )
        documents.append(doc)

    print(
        f"[document_loader] {len(documents)} blocos indexados | "
        f"{skipped_repeated} cabeçalhos/rodapés removidos | "
        f"{skipped_short} blocos curtos removidos"
    )
    return documents

