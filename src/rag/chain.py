from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM
import time

from config.settings import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, RETRIEVER_K


def get_conversation_chain(vector_db):
    llm = OllamaLLM(
        model=OLLAMA_LLM_MODEL,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    template = """És um analista de documentos inteligente.
Responde à pergunta usando apenas o contexto fornecido.
Se a resposta não estiver no contexto, diz explicitamente que a informação não foi encontrada.
Não inventes valores, datas, nomes ou conclusões.

Contexto:
{context}

Pergunta:
{question}

Resposta em português, clara e objetiva. Se houver tabelas no contexto, formata-as em Markdown:"""

    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    def _run(input_dict):
        question = input_dict["question"]
        t0 = time.perf_counter()
        source_docs = retriever.invoke(question)
        t1 = time.perf_counter()
        context = "\n\n".join(
            f"[Fonte: {doc.metadata.get('source')} | Página: {doc.metadata.get('page_number')} | "
            f"Bloco: {doc.metadata.get('block_id')}]\n{doc.page_content}"
            for doc in source_docs
        )

        answer = (prompt | llm | parser).invoke(
            {
                "context": context,
                "question": question,
            }
        )

        t2 = time.perf_counter()

        print(
            f"[RAG timing] retrieval={t1 - t0:.2f}s | "
            f"llm={t2 - t1:.2f}s | "
            f"total={t2 - t0:.2f}s",
            flush=True,
        )

        return {
            "answer": answer,
            "source_documents": source_docs,
            "timings": {
                "retrieval": round(t1 - t0, 2),
                "llm": round(t2 - t1, 2),
                "total": round(t2 - t0, 2),
            },
        }

    return RunnableLambda(_run)
