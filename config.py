import os

from langchain_community.document_loaders.base_o365 import CHUNK_SIZE

# ── Paths ──────────────────────────────────────
BOOKS_DIR = r"C:\Users\dhima\PycharmProjects\RAG_ML\DL tutor\data\books"
CHROMA_DIR = "chroma_db/"
LOG_FILE   = "logs/tutor_logs.log"

# ── Embedding Model ────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ── Text Splitter ──────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# ── Retriever ──────────────────────────────────
MMR_K                = 3
MMR_FETCH_K          = 20
MMR_LAMBDA           = 0.7
SIMILARITY_THRESHOLD = 0.75

# ── LLM ───────────────────────────────────────
GROQ_MODEL  = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3
MAX_TOKENS  = 1024
TOP_P       = 0.9

# ── Memory ────────────────────────────────────
MEMORY_K = 5


