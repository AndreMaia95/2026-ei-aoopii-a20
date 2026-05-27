from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


def process_blocks_to_langchain_docs(
    blocks: list[dict[str, Any]],
    source_name: str = "documento",
) -> list[Document]:
    """Convert OCR blocks into LangChain documents with traceable metadata."""
    documents: list[Document] = []

    for i, block in enumerate(blocks):
        text = block.get("text", "").strip()
        if not text:
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

    return documents
