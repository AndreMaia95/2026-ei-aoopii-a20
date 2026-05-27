from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config.settings import (
    ALLOW_DANGEROUS_DESERIALIZATION,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    VECTOR_INDEX_DIR,
)


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def create_or_load_vectorstore(
    documents: list[Document] | None = None,
    index_path: str | Path | None = None,
) -> FAISS | None:
    """Create a FAISS index from documents or load an existing local index."""
    target_path = Path(index_path) if index_path else VECTOR_INDEX_DIR
    embeddings = _get_embeddings()

    if documents:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        vector_db = FAISS.from_documents(documents, embeddings)
        vector_db.save_local(str(target_path))
        return vector_db

    if not target_path.exists():
        return None

    try:
        return FAISS.load_local(
            str(target_path),
            embeddings,
            allow_dangerous_deserialization=ALLOW_DANGEROUS_DESERIALIZATION,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Existe um índice FAISS local, mas o carregamento com pickle está bloqueado. "
            "Define ALLOW_DANGEROUS_DESERIALIZATION=true apenas para índices criados "
            "localmente e em que confias."
        ) from exc
