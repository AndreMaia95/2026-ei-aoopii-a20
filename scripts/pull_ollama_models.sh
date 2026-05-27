#!/usr/bin/env sh
set -eu

LLM_MODEL="${OLLAMA_LLM_MODEL:-llama3}"
EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"

ollama pull "$LLM_MODEL"
ollama pull "$EMBEDDING_MODEL"