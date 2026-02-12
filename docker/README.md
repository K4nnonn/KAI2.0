# Docker app run
1) Copy .env.example to .env and fill values
2) From /docker: docker compose up --build
3) API: http://localhost:8000
4) UI:  http://localhost:5173

## Local LLM (Ollama + Qwen)
Docker compose includes an Ollama service and will pull the configured model on first start.

Docker:
1) Set in `.env`:
   - `LOCAL_LLM_ENDPOINT=http://ollama:11434`
   - `LOCAL_LLM_MODEL=qwen2.5:14b`
2) `docker compose up --build`

Local install (no Docker):
1) Install Ollama from https://ollama.com
2) Pull a model: `ollama pull qwen2.5:14b` (or your preferred Qwen model)
3) Set:
   - `LOCAL_LLM_ENDPOINT=http://localhost:11434`
   - `LOCAL_LLM_MODEL=qwen2.5:14b`
