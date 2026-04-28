from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def get_conversation_chain(vector_db):
    llm = Ollama(model="llama3", temperature=0)  # Temp 0 para evitar alucinações
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    template = """És um analista de documentos inteligente. Responde à pergunta usando apenas o contexto fornecido.
Se a resposta não estiver no contexto, diz explicitamente que a informação não foi encontrada.

Contexto: {context}
Pergunta: {question}

Resposta (Se houver tabelas no contexto, formata-as em Markdown):"""

    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    def _run(input_dict):
        question = input_dict["question"]
        # Recuperar documentos relevantes (XAi: guardamos as fontes)
        source_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        # Gerar resposta
        answer = (prompt | llm | parser).invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": answer,
            "source_documents": source_docs
        }

    return RunnableLambda(_run)