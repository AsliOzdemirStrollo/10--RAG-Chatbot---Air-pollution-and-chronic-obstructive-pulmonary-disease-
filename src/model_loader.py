import os
from dotenv import load_dotenv

# LLM (GoogleGenAI)
from llama_index.llms.google_genai import GoogleGenAI

# Embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    LLM_MODEL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_CACHE_PATH,
    SEMANTIC_SPLITTER_EMBEDDING_MODEL_NAME,
    HYDE_LLM_MODEL_NAME,  # 👈 NEW: separate model name for HyDE (optional)
)

# Load environment variables from .env
load_dotenv()


# -------------------------------
#  LLM Loader (Main Chatbot LLM)
# -------------------------------
def initialise_llm() -> GoogleGenAI:
    """Initialises the main GoogleGenAI LLM using API key + model from config."""
    api_key: str | None = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Set it in your .env file.")

    return GoogleGenAI(
        api_key=api_key,
        model=LLM_MODEL,
    )


# -------------------------------
#  LLM Loader for HyDE
# -------------------------------
def initialise_hyde_llm() -> GoogleGenAI:
    """
    Initialises the LLM used specifically for HyDE synthetic answers.

    You can configure HYDE_LLM_MODEL_NAME in config.py. It can be the same
    as LLM_MODEL or a smaller/faster model.
    """
    api_key: str | None = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Set it in your .env file.")

    return GoogleGenAI(
        api_key=api_key,
        model=HYDE_LLM_MODEL_NAME,
    )


# -------------------------------
#  Embedding Model Loader (for RAG chunks)
# -------------------------------
def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model for RAG chunks."""

    # Create the embedding cache directory if it doesn't exist
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix(),
    )


# -------------------------------
#  Embedding Model Loader (for semantic splitter)
# -------------------------------
def get_splitter_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model for sentence embedding."""

    # Create the cache directory if it doesn't exist
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=SEMANTIC_SPLITTER_EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix(),
    )