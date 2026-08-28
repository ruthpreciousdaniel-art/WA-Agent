from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL, TOP_K, load_system_prompt
from app.rag.embeddings import embed_query
from app.rag.vector_store import search

_client = Groq(api_key=GROQ_API_KEY)

def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context was found in the knowledge base."
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c['source']}, relevance: {c['score']:.2f})\n{c['text']}")
    return "\n\n".join(parts)

def answer_query(query: str, top_k: int | None = None) -> dict:
    k = top_k or TOP_K
    query_vec = embed_query(query)
    retrieved = search(query_vec, top_k=k)
    context = build_context(retrieved)

    system_prompt = load_system_prompt()
    user_message = (
        f"Context from knowledge base:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Answer using only the context above. If the context doesn't contain "
        "the answer, say so clearly instead of guessing."
    )

    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    answer = completion.choices[0].message.content
    return {
        "answer": answer,
        "sources": [{"source": c["source"], "score": c["score"]} for c in retrieved],
    }
