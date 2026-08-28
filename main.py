import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.config import UPLOAD_DIR
from app.rag.document_loader import extract_text_from_pdf, chunk_text
from app.rag.vector_store import add_document, stats
from app.rag.agent import answer_query

app = FastAPI(title="RAG Agent API")


class ChatRequest(BaseModel):
    query: str
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", **stats()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    result = answer_query(req.query, top_k=req.top_k)
    return result


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest_path = Path(UPLOAD_DIR) / safe_name

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(str(dest_path))
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from this PDF.")

    chunks = chunk_text(text)
    added = add_document(source_name=file.filename, chunks=chunks)

    return {
        "message": f"Uploaded and indexed '{file.filename}'.",
        "chunks_added": added,
        "knowledge_base_stats": stats(),
    }
