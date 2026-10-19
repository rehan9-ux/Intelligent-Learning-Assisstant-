import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "ila_db"
SQLITE_DB_PATH = DATA_DIR / "ila_students.db"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# Ollama Offline Service Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:8b")

# RAG & Chunking Parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RESULTS = 4
MAX_DISTANCE_THRESHOLD = 1.5

# Adaptive Learning & BKT Parameters
DEFAULT_MASTERY_THRESHOLD = 0.75
WEAK_TOPIC_RATIO = 0.70  # 70% weak topic questions, 30% review questions
MIN_ATTEMPTS_FOR_BKT = 3
DEFAULT_STUDENT_ID = "student_01"
