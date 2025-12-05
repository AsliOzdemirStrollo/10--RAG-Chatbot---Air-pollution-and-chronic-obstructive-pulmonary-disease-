# --- Imports ---
from pathlib import Path


# --- LLM Model Configuration ---
LLM_MODEL: str = "gemini-2.5-flash"
LLM_MAX_NEW_TOKENS: int = 768
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03

# System prompt defining chatbot personality
LLM_SYSTEM_PROMPT: str = (
    "You are a helpful chatbot. Be friendly and conversational."
)


# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- Semantic Splitter Configuration ---
# Embedding model used for sentence-level semantic splitting
SEMANTIC_SPLITTER_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# How many neighbouring sentences to consider when merging
BUFFER_SIZE: int = 1  # 1 is usually fine to start

# How strict we are about topic similarity (1–100, higher = more splits)
BREAKPOINT_PERCENTILE_THRESHOLD: int = 90


# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
# (Updated to 10 based on evaluation results)
SIMILARITY_TOP_K: int = 10

# The size of each text chunk in tokens (approx words is okay)
CHUNK_SIZE: int = 512

# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 80


# --- Reranker Configuration ---
# How many chunks to keep after reranking
RERANKER_TOP_N: int = 5

# SentenceTransformer reranker model name
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"


# --- HyDE LLM Configuration (Optional) ---
# You can reuse the main LLM for HyDE, or later switch this to a smaller model if you like
HYDE_LLM_MODEL_NAME: str = "gemini-2.5-flash"


# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage" / "embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage" / "vector_store/"