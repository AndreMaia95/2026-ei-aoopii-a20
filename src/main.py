from __future__ import annotations

import streamlit as st

from config.settings import ALLOWED_FILE_TYPES, APP_SUBTITLE, APP_TITLE, VECTOR_INDEX_DIR
from ocr.preprocessing import preprocess_for_ocr, segment_and_ocr, validate_tesseract
from rag.chain import get_conversation_chain
from rag.document_loader import process_blocks_to_langchain_docs
from rag.vectorstore import create_or_load_vectorstore
from utils.visualizer import confidence_color, confidence_label, render_page_with_highlight


st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧠")

# ─────────────────────────────────────────────
# CSS GLOBAL — Dark Mode Premium
# ─────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-weight: 600; }
[data-testid="stSidebar"] .stButton > button { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; padding: 0.55rem 1rem !important; transition: all 0.2s ease !important; width: 100% !important; }
[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important; }
[data-testid="stFileUploader"] { background: #1e2535 !important; border: 1.5px dashed #334155 !important; border-radius: 12px !important; padding: 8px !important; }
.main .block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }
[data-testid="stChatMessage"] { background: #1a2035 !important; border: 1px solid #1e2d45 !important; border-radius: 16px !important; padding: 0.8rem 1.2rem !important; margin-bottom: 0.5rem !important; }
[data-testid="stExpander"] { background: #141928 !important; border: 1px solid #1e2d45 !important; border-radius: 14px !important; overflow: hidden !important; }
[data-testid="stExpander"] summary { color: #a5b4fc !important; font-weight: 600 !important; font-size: 0.9rem !important; }
[data-testid="stExpander"] summary:hover { color: #c4b5fd !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }
div[data-testid="stAlert"][data-baseweb="notification"] { background: #1e2d45 !important; border-left-color: #6366f1 !important; color: #cbd5e1 !important; }
hr { border-color: #1e2535 !important; margin: 0.8rem 0 !important; }
[data-testid="stChatInput"] { background: #131929 !important; border: 1px solid #1e2d45 !important; border-radius: 12px !important; padding: 2px 4px !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; border: none !important; outline: none !important; box-shadow: none !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stChatInput"] textarea:focus { border: none !important; box-shadow: none !important; outline: none !important; }
[data-testid="stChatInput"]:focus-within { border-color: #334155 !important; box-shadow: none !important; }
[data-testid="stChatInputSubmitButton"] button { background: #1e2d45 !important; border: none !important; border-radius: 8px !important; color: #a5b4fc !important; }
[data-testid="stChatInputSubmitButton"] button:hover { background: #6366f1 !important; color: white !important; }
[data-testid="stSpinner"] { color: #a5b4fc !important; }
.stButton > button { background: #1e2535 !important; color: #a5b4fc !important; border: 1px solid #334155 !important; border-radius: 8px !important; font-weight: 500 !important; transition: all 0.2s ease !important; }
.stButton > button:hover { border-color: #6366f1 !important; color: #c4b5fd !important; background: #1e2a45 !important; }
img { border-radius: 10px !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.5rem;
">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="
            width: 40px; height: 40px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 10px;
            flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
        ">🧠</div>
        <div>
            <div style="
                font-size: 1.5rem; font-weight: 700; line-height: 1.2;
                background: linear-gradient(90deg, #a5b4fc, #c4b5fd);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            ">Document Intelligence</div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 2px; letter-spacing: 0.3px;">
                RAG &nbsp;·&nbsp; OCR &nbsp;·&nbsp; Explainable AI
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # -- Secção: Upload --
    st.markdown("""
    <div style="font-size:0.7rem;font-weight:600;color:#475569;
                letter-spacing:1px;text-transform:uppercase;
                margin-bottom:0.6rem;">Documento</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Carrega um PDF ou imagem",
        type=ALLOWED_FILE_TYPES,
        label_visibility="collapsed",
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    process_btn = st.button("Processar", use_container_width=True)

    # -- Secção: Estado do documento --
    st.markdown("""
    <div style="height:1px;background:#1e2535;margin:1.2rem 0;"></div>
    <div style="font-size:0.7rem;font-weight:600;color:#475569;
                letter-spacing:1px;text-transform:uppercase;
                margin-bottom:0.8rem;">Estado</div>
    """, unsafe_allow_html=True)

    # O painel de estado é preenchido depois de termos o session_state


# ─────────────────────────────────────────────
# ESTADO DA SESSÃO
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "processed_document" not in st.session_state:
    st.session_state.processed_document = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}


# ─────────────────────────────────────────────
# PROCESSAMENTO DO DOCUMENTO
# ─────────────────────────────────────────────
if uploaded_file and process_btn:
    st.session_state.vector_db = None
    st.session_state.processed_document = None
    st.session_state.chat_history = []
    st.session_state.doc_stats = {}
    st.session_state.pdf_bytes = uploaded_file.read() if uploaded_file.name.lower().endswith(".pdf") else None
    uploaded_file.seek(0)

    with st.spinner("A processar documento com OCR e a indexar no FAISS…"):
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
                st.session_state.doc_stats = {
                    "pages": len(pages),
                    "blocks": len(all_blocks),
                    "indexed": len(docs),
                    "tesseract": tesseract_version,
                }
        except Exception as exc:
            st.error(f"Falha ao processar o documento: {exc}")


# ─────────────────────────────────────────────
# PAINEL DE ESTADO NA SIDEBAR (preenchido com session_state)
# ─────────────────────────────────────────────
with st.sidebar:
    stats = st.session_state.doc_stats
    doc_name = st.session_state.processed_document

    if doc_name:
        # Documento ativo
        st.markdown(f"""
        <div style="
            background:#0f1117; border:1px solid #1e2d45;
            border-radius:10px; padding:0.8rem 0.9rem;
        ">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.6rem;">
                <div style="width:7px;height:7px;border-radius:50%;background:#22c55e;flex-shrink:0;"></div>
                <span style="font-size:0.75rem;color:#22c55e;font-weight:600;">Ativo</span>
            </div>
            <div style="font-size:0.82rem;color:#e2e8f0;font-weight:500;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                        margin-bottom:0.7rem;" title="{doc_name}">
                {doc_name}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
                <div style="background:#161b27;border-radius:7px;padding:6px 8px;">
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:2px;">Páginas</div>
                    <div style="font-size:1rem;font-weight:700;color:#a5b4fc;">
                        {stats.get('pages', '—')}
                    </div>
                </div>
                <div style="background:#161b27;border-radius:7px;padding:6px 8px;">
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:2px;">Blocos</div>
                    <div style="font-size:1rem;font-weight:700;color:#a5b4fc;">
                        {stats.get('indexed', '—')}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        # Sem documento
        st.markdown("""
        <div style="
            background:#0f1117; border:1px dashed #1e2535;
            border-radius:10px; padding:0.9rem;
            text-align:center;
        ">
            <div style="font-size:0.8rem;color:#334155;line-height:1.5;">
                Nenhum documento carregado.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -- Rodapé da sidebar --
    st.markdown("""
    <div style="
        margin-top: 2rem;
        font-size: 0.68rem; color: #1e2d45;
        border-top: 1px solid #1e2535; padding-top: 0.7rem;
        letter-spacing: 0.3px;
    ">
        FAISS &nbsp;·&nbsp; Tesseract OCR &nbsp;·&nbsp; LangChain
    </div>
    """, unsafe_allow_html=True)

# Linha separadora antes do chat
if st.session_state.processed_document:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LÓGICA XAi — (sem alterações)
# ─────────────────────────────────────────────
import re
import string


def _filter_relevant_sources(question: str, answer: str, sources: list) -> list:
    """Filtra as fontes confiando no FAISS, mas adicionando blocos extras se a resposta assim o exigir."""
    if not sources:
        return []

    stopwords = {"que", "não", "foi", "sobre", "como", "mas", "para", "com", "por", "qual", "quem", "uma", "um", "dos", "das", "sua", "área", "em", "de", "e", "a", "o", "os", "as", "é", "são", "tem"}

    def get_words(text: str):
        text = re.sub(r'(?<=[\d])\s+(?=[\d])', '', text)
        for p in string.punctuation:
            text = text.replace(p, ' ')
        words = text.lower().split()
        return set(w for w in words if len(w) > 2 and w not in stopwords)

    def get_numbers(word_set):
        return {w for w in word_set if w.isdigit()}

    q_words = get_words(question)
    ans_words = get_words(answer)

    target_words = ans_words - q_words
    if not target_words:
        target_words = ans_words

    results = []

    for _ in range(3):
        if not target_words:
            break

        best_doc = None
        best_overlap = set()

        for doc in sources:
            if doc in results:
                continue
            doc_words = get_words(doc.page_content)
            overlap = target_words & doc_words

            def calc_score(ov_set):
                return sum(100 if w.isdigit() else 1 for w in ov_set)

            score = calc_score(overlap)
            best_score = calc_score(best_overlap)

            if score > best_score:
                best_overlap = overlap
                best_doc = doc
            elif score == best_score and score > 0:
                conf1 = doc.metadata.get("confidence", 0)
                conf2 = best_doc.metadata.get("confidence", 0) if best_doc else 0
                if conf1 > conf2:
                    best_overlap = overlap
                    best_doc = doc

        if best_doc:
            score = calc_score(best_overlap)
            if score >= 100 or score >= 3 or len(results) == 0:
                results.append(best_doc)
                target_words -= best_overlap
            else:
                break
        else:
            break

    if not results:
        return sources[:1]

    results.sort(key=lambda d: (d.metadata.get("page_number", 0), d.metadata.get("block_id", 0)))
    return _merge_consecutive_blocks(results)


def _merge_consecutive_blocks(docs: list) -> list:
    """Funde blocos consecutivos (mesma página, block_id seguido) num único Document."""
    if len(docs) <= 1:
        return docs

    from langchain_core.documents import Document

    merged = []
    i = 0
    while i < len(docs):
        current = docs[i]
        if i + 1 < len(docs):
            nxt = docs[i + 1]
            curr_page = current.metadata.get("page_number")
            nxt_page = nxt.metadata.get("page_number")
            curr_block = current.metadata.get("block_id", -99)
            nxt_block = nxt.metadata.get("block_id", -999)

            if curr_page == nxt_page and (nxt_block - curr_block) <= 2:
                merged_text = current.page_content.rstrip() + " " + nxt.page_content.lstrip()

                c1 = current.metadata.get("coords")
                c2 = nxt.metadata.get("coords")
                merged_coords = c1
                if c1 and c2:
                    try:
                        x0 = min(c1.get("x", 0), c2.get("x", 0))
                        y0 = min(c1.get("y", 0), c2.get("y", 0))
                        x1 = max(c1.get("x", 0) + c1.get("w", 0), c2.get("x", 0) + c2.get("w", 0))
                        y1 = max(c1.get("y", 0) + c1.get("h", 0), c2.get("y", 0) + c2.get("h", 0))
                        merged_coords = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
                    except (AttributeError, TypeError):
                        merged_coords = c1

                conf1 = current.metadata.get("confidence") or 0
                conf2 = nxt.metadata.get("confidence") or 0
                len1 = len(current.page_content)
                len2 = len(nxt.page_content)
                merged_conf = (conf1 * len1 + conf2 * len2) / (len1 + len2) if (len1 + len2) > 0 else conf1

                fused_doc = Document(
                    page_content=merged_text,
                    metadata={
                        **current.metadata,
                        "coords": merged_coords,
                        "confidence": round(merged_conf, 1),
                    }
                )
                merged.append(fused_doc)
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged


def _render_xai_panel(source_docs: list) -> None:
    """Painel XAi visual — dark mode premium."""
    if not source_docs:
        st.markdown("""
        <div style="color:#475569;padding:0.8rem;text-align:center;font-size:0.875rem;">
            Sem evidências encontradas.
        </div>""", unsafe_allow_html=True)
        return

    # Badge de páginas consultadas
    pages_used = sorted({
        doc.metadata.get("page_number")
        for doc in source_docs
        if doc.metadata.get("page_number")
    })
    if pages_used:
        badges = "".join([
            f'<span style="background:#1e2d45;color:#a5b4fc;border:1px solid #334155;'
            f'border-radius:6px;padding:2px 10px;font-size:0.75rem;font-weight:600;margin-right:6px;">'
            f'Pág. {p}</span>'
            for p in pages_used
        ])
        st.markdown(
            f'<div style="margin-bottom:12px;">'
            f'<span style="color:#64748b;font-size:0.8rem;margin-right:8px;">Páginas consultadas:</span>'
            f'{badges}</div>',
            unsafe_allow_html=True,
        )

    for i, doc in enumerate(source_docs):
        page_num = doc.metadata.get("page_number")
        coords = doc.metadata.get("coords")
        confidence = doc.metadata.get("confidence") or 0

        color = confidence_color(confidence)
        label = confidence_label(confidence)

        if i > 0:
            st.markdown("<hr style='border-color:#1e2535;margin:12px 0;'>", unsafe_allow_html=True)

        col_img, col_info = st.columns([1, 1])

        # Coluna esquerda: preview do PDF
        with col_img:
            if st.session_state.pdf_bytes and page_num and coords:
                try:
                    img = render_page_with_highlight(
                        st.session_state.pdf_bytes, page_num, coords
                    )
                    st.markdown("""
                    <div style="
                        border: 1px solid #1e2d45;
                        border-radius: 10px;
                        overflow: hidden;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                    ">""", unsafe_allow_html=True)
                    st.image(img, caption=f"Página {page_num}", width="stretch")
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div style="color:#475569;font-size:0.8rem;">Página {page_num}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:#475569;font-size:0.8rem;">Página {page_num}</div>', unsafe_allow_html=True)

        # Coluna direita: confiança + trecho
        with col_info:
            # Barra de confiança animada
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-size:0.8rem;color:#94a3b8;font-weight:500;">Confiança OCR</span>
                    <span style="font-size:0.8rem;font-weight:700;color:{color};">
                        {label} &nbsp;{confidence:.1f}%
                    </span>
                </div>
                <div style="background:#1e2535;border-radius:99px;height:8px;overflow:hidden;">
                    <div style="
                        width:{confidence}%;height:8px;border-radius:99px;
                        background:linear-gradient(90deg,{color}88,{color});
                        transition:width 0.6s ease;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Trecho extraído — estilo "marcador de texto"
            st.markdown("""
            <div style="font-size:0.78rem;color:#64748b;font-weight:600;
                        letter-spacing:0.5px;margin-bottom:6px;text-transform:uppercase;">
                Trecho extraído
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="
                background:#1e2d45;
                border-left: 3px solid #6366f1;
                border-radius: 0 10px 10px 0;
                padding: 0.75rem 1rem;
                color: #e2e8f0;
                font-size: 0.92rem;
                line-height: 1.6;
                font-style: italic;
            ">{doc.page_content}</div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ÁREA DE CHAT
# ─────────────────────────────────────────────
if st.session_state.vector_db:
    # Histórico anterior
    for user_msg, assistant_msg, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
        with st.expander("Ver evidências — XAi"):
            _render_xai_panel(sources)

    user_question = st.chat_input("Faz uma pergunta sobre o documento…")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("A processar…"):
                chain = get_conversation_chain(st.session_state.vector_db)
                response = chain.invoke(
                    {
                        "question": user_question,
                        "chat_history": [
                            (u, a) for u, a, _ in st.session_state.chat_history
                        ],
                    }
                )
            st.markdown(response["answer"])

            relevant_sources = _filter_relevant_sources(
                user_question, response["answer"], response["source_documents"]
            )

        with st.expander("Ver evidências — XAi", expanded=True):
            _render_xai_panel(relevant_sources)

        st.session_state.chat_history.append(
            (user_question, response["answer"], relevant_sources)
        )
        st.rerun()

else:
    # Landing zone
    st.markdown("""
    <div style="
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; text-align: center;
        padding: 4rem 2rem; margin-top: 2rem;
        background: #0d1117;
        border: 1px solid #1e2535;
        border-radius: 16px;
    ">
        <div style="
            width: 56px; height: 56px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 14px;
            margin: 0 auto 1.5rem;
            display: flex; align-items: center; justify-content: center;
            font-size: 26px;
        ">🧠</div>
        <div style="
            font-size: 1.2rem; font-weight: 700; color: #e2e8f0;
            margin-bottom: 0.6rem;
        ">Nenhum documento carregado</div>
        <div style="color: #475569; font-size: 0.875rem; max-width: 340px; line-height: 1.7;">
            Carrega um PDF ou imagem no painel lateral e clica em
            <strong style="color:#94a3b8;">Processar</strong>
            para começar a fazer perguntas.
        </div>
        <div style="
            display: flex; gap: 10px; margin-top: 2rem; flex-wrap: wrap;
            justify-content: center;
        ">
            <span style="background:#161b27;color:#6366f1;border:1px solid #1e2d45;
                border-radius:6px;padding:5px 14px;font-size:0.75rem;font-weight:600;
                letter-spacing:0.5px;">OCR</span>
            <span style="background:#161b27;color:#8b5cf6;border:1px solid #1e2d45;
                border-radius:6px;padding:5px 14px;font-size:0.75rem;font-weight:600;
                letter-spacing:0.5px;">RAG + FAISS</span>
            <span style="background:#161b27;color:#a78bfa;border:1px solid #1e2d45;
                border-radius:6px;padding:5px 14px;font-size:0.75rem;font-weight:600;
                letter-spacing:0.5px;">Explainable AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
