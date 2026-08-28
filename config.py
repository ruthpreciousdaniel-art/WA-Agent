import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
SYSTEM_PROMPT_PATH = APP_DIR / "system_prompt.txt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it as an environment variable / platform secret."
    )

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800)
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 120)
TOP_K = int(os.environ.get("TOP_K", 4))

def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text()
    return "You are a helpful RAG assistant. Answer using only the provided context."
