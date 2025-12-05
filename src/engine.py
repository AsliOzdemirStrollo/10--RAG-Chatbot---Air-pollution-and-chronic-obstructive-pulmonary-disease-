# src/engine.py


from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Retrieval + HyDE + Reranker
from llama_index.core.retrievers import BaseRetriever, TransformRetriever
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.config import (
    CHUNK_OVERLAP,  # still imported even if not used directly here
    CHUNK_SIZE,     # same
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
    BUFFER_SIZE,
    BREAKPOINT_PERCENTILE_THRESHOLD,
    RERANKER_TOP_N,
    RERANKER_MODEL_NAME,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm,
    get_splitter_embedding_model,
    initialise_hyde_llm,
)


def _create_new_vector_store(
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""
    print("Creating new vector store from all files in the 'data' directory...")

    documents: list[Document] = SimpleDirectoryReader(
        input_files=[DATA_PATH / "air_pollution.txt"]
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # ---------------------------------------------------------
    # Semantic Splitter
    # ---------------------------------------------------------
    semantic_splitter_embedding_model = get_splitter_embedding_model()

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=BUFFER_SIZE,
        breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE_THRESHOLD,
        embed_model=semantic_splitter_embedding_model,
    )
    # ---------------------------------------------------------

    # Create vector store using semantic splitter
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[semantic_splitter],
        embed_model=embed_model,
    )

    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved using Semantic Splitter.")
    return index


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """
    # Create the parent directory if it doesn't exist.
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Check if the directory contains any files.
    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        # We must provide the embed_model when loading the index.
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )
    else:
        # If the directory is empty,
        # call our internal function to build the index.
        return _create_new_vector_store(embed_model)


def get_chat_engine(
    llm: GoogleGenAI,
    embed_model: HuggingFaceEmbedding,
) -> CondensePlusContextChatEngine:
    """Initialises and returns the main conversational RAG chat engine."""

    # 1. Access existing vector index (saved vector store)
    vector_index: VectorStoreIndex = get_vector_store(embed_model)

    # 2. Base retriever from index
    base_retriever: BaseRetriever = vector_index.as_retriever(
        similarity_top_k=SIMILARITY_TOP_K
    )

    # 3. HyDE query transform (synthetic hypothetical answers)
    hyde: HyDEQueryTransform = HyDEQueryTransform(
        include_original=True,
        llm=initialise_hyde_llm(),  # you can swap to initialise_llm() if you prefer
    )

    # 4. Wrap the retriever with HyDE
    hyde_retriever: TransformRetriever = TransformRetriever(
        retriever=base_retriever,
        query_transform=hyde,
    )

    # 5. Reranker (SentenceTransformerRerank – same idea as evaluation)
    reranker: SentenceTransformerRerank = SentenceTransformerRerank(
        top_n=RERANKER_TOP_N,
        model=RERANKER_MODEL_NAME,
    )

    # 6. Memory for conversation (used for query condensing)
    memory: ChatSummaryMemoryBuffer = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    # 7. Chat engine with:
    #    - query condensation (rewrite)
    #    - HyDE retriever
    #    - reranker
    chat_engine: CondensePlusContextChatEngine = CondensePlusContextChatEngine(
        retriever=hyde_retriever,
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT,
        node_postprocessors=[reranker],
    )

    return chat_engine


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def main_chat_loop() -> None:
    """Main application loop to run the RAG chatbot."""
    print("--- Initialising models... ---")

    llm: GoogleGenAI = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: CondensePlusContextChatEngine = get_chat_engine(
        llm=llm,
        embed_model=embed_model,
    )

    print("--- RAG Chatbot Initialised. ---")
    chat_engine.chat_repl()