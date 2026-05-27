from __future__ import annotations

import streamlit as st

from config.settings import ALLOWED_FILE_TYPES, APP_SUBTITLE, APP_TITLE, VECTOR_INDEX_DIR
from ocr.preprocessing import preprocess_for_ocr, segment_and_ocr, validate_tesseract
from rag.chain import get_conversation_chain
from rag.document_loader import process_blocks_to_langchain_docs
from rag.vectorstore import create_or_load_vectorstore


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)

with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader(
        "Carrega um documento (PDF ou imagem)",
        type=ALLOWED_FILE_TYPES,
    )
    process_btn = st.button("Processar documento")
    st.caption(f"Índice FAISS: `{VECTOR_INDEX_DIR}`")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "processed_document" not in st.session_state:
    st.session_state.processed_document = None

if uploaded_file and process_btn:
    st.session_state.vector_db = None
    st.session_state.processed_document = None
    st.session_state.chat_history = []

    with st.spinner("A validar Tesseract, processar OCR e indexar o documento..."):
        try:
            tesseract_version = validate_tesseract()
            pages = preprocess_for_ocr(uploaded_file)

            all_blocks = []
            for page in pages:
                blocks = segment_and_ocr(
                    page["processed_img"],
                    page_number=page["page_number"],
                )
                all_blocks.extend(blocks)

            docs = process_blocks_to_langchain_docs(all_blocks, source_name=uploaded_file.name)
            if not docs:
                st.warning("Não foi possível extrair texto útil deste documento.")
            else:
                st.session_state.vector_db = create_or_load_vectorstore(docs)
                st.session_state.chat_history = []
                st.session_state.processed_document = uploaded_file.name
                st.success(
                    f"Documento pronto: {len(pages)} página(s), "
                    f"{len(all_blocks)} bloco(s), {len(docs)} documento(s) indexado(s). "
                    f"Tesseract: {tesseract_version}."
                )
        except Exception as exc:
            st.error(f"Falha ao processar o documento: {exc}")

if st.session_state.processed_document:
    st.info(f"Documento ativo: **{st.session_state.processed_document}**")

if st.session_state.vector_db:
    user_question = st.chat_input("Faz uma pergunta sobre o documento...")

    if user_question:
        chain = get_conversation_chain(st.session_state.vector_db)
        with st.spinner("A procurar evidências no documento e gerar resposta..."):
            response = chain.invoke(
                {
                    "question": user_question,
                    "chat_history": st.session_state.chat_history,
                }
            )

        st.session_state.chat_history.append((user_question, response["answer"]))

        with st.chat_message("assistant"):
            st.markdown(response["answer"])

            with st.expander("Ver evidências"):
                for doc in response["source_documents"]:
                    st.write(f"**Trecho:** {doc.page_content}")
                    st.write(f"**Fonte:** {doc.metadata.get('source')}")
                    st.write(f"**Página:** {doc.metadata.get('page_number')}")
                    st.write(f"**Bloco:** {doc.metadata.get('block_id')}")
                    st.write(f"**Confiança OCR:** {doc.metadata.get('confidence')}")
                    st.write(f"**Coordenadas:** {doc.metadata.get('coords')}")
                    st.divider()
else:
    st.info("Carrega e processa um documento para iniciar o chat.")
