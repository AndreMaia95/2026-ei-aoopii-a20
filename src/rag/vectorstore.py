from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_or_load_vectorstore(documents=None, index_path="faiss_index"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if documents:
        # Criar nova base de dados
        vector_db = FAISS.from_documents(documents, embeddings)
        vector_db.save_local(index_path)
        return vector_db
    elif os.path.exists(index_path):
        # Carregar existente
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return None