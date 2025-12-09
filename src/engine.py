# src/engine.py

from __future__ import annotations

import sys

from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.schema import Document

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from src.config import (
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
)

from src.model_loader import (
    get_embedding_model,
    initialise_llm,
)


def _log(msg: str) -> None:
    """Small helper to log to Streamlit Cloud logs."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------
# Vector store creation / loading (simplified: no semantic splitter)
# ---------------------------------------------------------------------
def _create_new_vector_store(
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""
    _log("engine: creating new SIMPLE vector store from air_pollution.txt ...")

    documents: list[Document] = SimpleDirectoryReader(
        input_files=[DATA_PATH / "air_pollution.txt"]
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # Simple index build: no semantic splitter, just default chunking
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
    )

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())

    _log("engine: SIMPLE vector store created and saved.")
    return index


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if any(VECTOR_STORE_PATH.iterdir()):
        _log("engine: loading existing vector store from disk ...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )

    _log("engine: no stored vector index found, creating a new one ...")
    return _create_new_vector_store(embed_model)


# ---------------------------------------------------------------------
# Chat engine (deployment-friendly)
# ---------------------------------------------------------------------
def get_chat_engine(
    llm: GoogleGenAI,
    embed_model: HuggingFaceEmbedding,
) -> CondensePlusContextChatEngine:
    """
    Initialises and returns the main conversational RAG chat engine.

    Deployment-friendly version:
    - Uses a standard retriever from the vector store
    - Uses Condense+Context chat engine
    - Keeps conversation memory
    - No HyDE, no reranker, no semantic splitter
    """
    _log("engine: get_chat_engine() start")

    # 1. Vector index
    vector_index: VectorStoreIndex = get_vector_store(embed_model)
    _log("engine: vector index ready")

    # 2. Base retriever
    base_retriever = vector_index.as_retriever(
        similarity_top_k=SIMILARITY_TOP_K
    )
    _log("engine: base retriever ready")

    # 3. Conversation memory (for query condensing)
    memory: ChatSummaryMemoryBuffer = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )
    _log("engine: memory buffer ready")

    # 4. Chat engine
    chat_engine: CondensePlusContextChatEngine = CondensePlusContextChatEngine(
        retriever=base_retriever,
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT,
        node_postprocessors=[],
    )

    _log("engine: chat engine created")
    return chat_engine


# ---------------------------------------------------------------------
# MAIN LOOP (for local CLI testing)
# ---------------------------------------------------------------------
def main_chat_loop() -> None:
    """Main application loop to run the RAG chatbot."""
    _log("--- Initialising models (CLI mode) ---")

    llm: GoogleGenAI = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: CondensePlusContextChatEngine = get_chat_engine(
        llm=llm,
        embed_model=embed_model,
    )

    _log("--- RAG Chatbot Initialised (CLI). ---")
    chat_engine.chat_repl()