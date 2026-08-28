# RAG Agent (FastAPI + FAISS-CPU + Groq)

## Endpoints
- GET  /health         -> index stats
- POST /chat           -> {"query": "..."} -> {"answer": "...", "sources": [...]}
- POST /upload         -> multipart form file=<pdf>

## Deploy on Render / Railway
1. Push this folder to a GitHub repo.
2. Create a new Web Service, point it at the repo, choose "Docker" as the environment.
3. Set env var GROQ_API_KEY (and optionally GROQ_MODEL, EMBEDDING_MODEL, TOP_K).
4. Both platforms inject $PORT automatically; the Dockerfile CMD already uses it.
5. IMPORTANT: default disk is ephemeral. Attach a persistent volume mounted at
   /code/data if you need the FAISS index to survive redeploys.
