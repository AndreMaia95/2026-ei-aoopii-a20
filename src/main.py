import streamlit as st
from ocr.preprocessing import preprocess_for_ocr, segment_and_ocr
from rag.document_loader import process_blocks_to_langchain_docs
from rag.vectorstore import create_or_load_vectorstore
from rag.chain import get_conversation_chain

st.set_page_config(page_title="Document Intelligence XAi", layout="wide")

st.title("Document Intelligence com XAi")
st.subheader("Extração com Tesseract + RAG Local (Ollama)")

# Sidebar para Upload
with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader("Carrega um documento (PDF ou Imagem)", type=["pdf", "png", "jpg", "jpeg"])
    process_btn = st.button("Processar Documento")

# Inicializar Estado da Sessão (Memória do Chat)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Fluxo Principal
if uploaded_file and process_btn:
    with st.spinner("A processar OCR e a indexar..."):
        # 1. OCR & Segmentação (suporta PDF multi-página e imagens)
        pages = preprocess_for_ocr(uploaded_file)
        all_blocks = []
        for processed_img, _ in pages:
            blocks = segment_and_ocr(processed_img)
            all_blocks.extend(blocks)

        # 2. Conversão para Documentos Langchain com Metadados (XAi)
        docs = process_blocks_to_langchain_docs(all_blocks, source_name=uploaded_file.name)

        # 3. Vector Store
        st.session_state.vector_db = create_or_load_vectorstore(docs)
        st.success(f"Documento pronto! ({len(pages)} página(s), {len(all_blocks)} blocos extraídos)")


# Interface de Chat
if st.session_state.vector_db:
    user_question = st.chat_input("Faz uma pergunta sobre o documento...")
    
    if user_question:
        chain = get_conversation_chain(st.session_state.vector_db)
        response = chain.invoke({"question": user_question, "chat_history": st.session_state.chat_history})
        
        # Atualizar histórico
        st.session_state.chat_history.append((user_question, response["answer"]))
        
        # Mostrar Resposta com Fontes (Explainability)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            
            with st.expander("Ver Evidências (XAi)"):
                for doc in response["source_documents"]:
                    st.write(f"**Trecho:** {doc.page_content}")
                    st.write(f"**Metadados:** {doc.metadata}")
                    st.divider()