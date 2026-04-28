query = "Qual é o valor total desta fatura e a data de vencimento?"
result = qa_chain.invoke({"query": query})

print(f"Resposta: {result['result']}")

# XAi: Mostrar ao utilizador de onde veio a informação
print("\n--- Fontes Consultadas ---")
for doc in result["source_documents"]:
    print(f"Bloco ID: {doc.metadata.get('block_id')} | Coordenadas: {doc.metadata.get('coords')}")