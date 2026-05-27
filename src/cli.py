import argparse
import sys

from rag.chain import get_conversation_chain
from rag.vectorstore import create_or_load_vectorstore

def main():
    parser = argparse.ArgumentParser(description="Perguntar a um índice FAISS já criado.")
    parser.add_argument("question", help="Pergunta a fazer sobre o documento indexado.")
    parser.add_argument(
        "--index",
        default=None,
        help="Caminho do índice FAISS. Por omissão usa VECTOR_INDEX_DIR.",
    )
    args = parser.parse_args()

    vector_db = create_or_load_vectorstore(index_path=args.index)
    if vector_db is None:
        print("Não foi encontrado nenhum índice FAISS. Processa primeiro um documento na app.")
        return 1

    chain = get_conversation_chain(vector_db)
    result = chain.invoke({"question": args.question, "chat_history": []})

    print(f"Resposta:\n{result['answer']}\n")
    print("--- Fontes consultadas ---")
    for doc in result["source_documents"]:
        print(
            "Fonte: {source} | Página: {page} | Bloco: {block} | Coordenadas: {coords}".format(
                source=doc.metadata.get("source"),
                page=doc.metadata.get("page_number"),
                block=doc.metadata.get("block_id"),
                coords=doc.metadata.get("coords"),
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
