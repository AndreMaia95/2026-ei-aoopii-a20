from langchain_core.documents import Document

def process_blocks_to_langchain_docs(blocks, source_name="documento"):
    """
    Converte os blocos do OCR (imagem + coords) em Documentos Langchain.
    """
    documents = []
    for i, block in enumerate(blocks):
        # Aqui assumimos que o texto já foi extraído pelo Tesseract no passo anterior
        doc = Document(
            page_content=block['text'],
            metadata={
                "source": source_name,
                "block_id": i,
                "coords": block['coords'], # (x, y, w, h)
                "type": "text_block"
            }
        )
        documents.append(doc)
    return documents