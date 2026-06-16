#!/bin/sh
# entrypoint.sh — espera o Ollama estar pronto e garante que os modelos existem

set -e

OLLAMA_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
LLM_MODEL="${OLLAMA_LLM_MODEL:-llama3}"
EMBED_MODEL="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"

echo "[entrypoint] A aguardar Ollama em $OLLAMA_URL ..."
until curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
  sleep 2
done
echo "[entrypoint] Ollama disponível."

# Verificar/puxar modelo LLM
if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$LLM_MODEL\""; then
  echo "[entrypoint] A puxar modelo LLM: $LLM_MODEL ..."
  curl -sf -X POST "$OLLAMA_URL/api/pull" -d "{\"name\":\"$LLM_MODEL\"}" > /dev/null
  echo "[entrypoint] $LLM_MODEL pronto."
else
  echo "[entrypoint] $LLM_MODEL já existe."
fi

# Verificar/puxar modelo de embeddings
if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$EMBED_MODEL\""; then
  echo "[entrypoint] A puxar modelo de embeddings: $EMBED_MODEL ..."
  curl -sf -X POST "$OLLAMA_URL/api/pull" -d "{\"name\":\"$EMBED_MODEL\"}" > /dev/null
  echo "[entrypoint] $EMBED_MODEL pronto."
else
  echo "[entrypoint] $EMBED_MODEL já existe."
fi

echo "[entrypoint] Tudo pronto. A arrancar Streamlit..."
exec streamlit run src/main.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
