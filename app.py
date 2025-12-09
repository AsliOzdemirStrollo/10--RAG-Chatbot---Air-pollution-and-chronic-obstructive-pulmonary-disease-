# app.py

import streamlit as st

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage, MessageRole

from src.engine import get_chat_engine
from src.model_loader import initialise_llm, get_embedding_model


# Pretty names for source files
DOC_TITLES = {
    "air_pollution.pdf": (
        "Air pollution and chronic obstructive pulmonary disease (Review Article)"
    ),
    "Air pollution and chronic obstructive pulmonary disease.pdf": (
        "Air pollution and chronic obstructive pulmonary disease (Review Article)"
    ),
    "air_pollution.txt": (
        "Air pollution and chronic obstructive pulmonary disease (Review Article)"
    ),
}


def init_chat_engine() -> object:
    """
    TEMP FIX:
    Simple LLM-only chat engine with conversation context.
    Still no RAG, but uses chat history so follow-up questions make more sense.
    """
    llm: GoogleGenAI = initialise_llm()

    class SimpleChatEngine:
        def __init__(self, llm):
            self.llm = llm

        def chat(self, query: str):
            import streamlit as st

            messages: list[ChatMessage] = []

            # System prompt to keep it on-topic
            system_text = (
                "You are a helpful assistant that answers questions about "
                "air pollution and chronic obstructive pulmonary disease (COPD). "
                "Use the previous conversation turns to interpret short or ambiguous "
                "follow-up questions like 'what about in China?' or 'and there?'."
            )
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_text))

            # Add previous conversation from Streamlit chat history
            for msg in st.session_state.get("messages", []):
                role = (
                    MessageRole.USER
                    if msg["role"] == "user"
                    else MessageRole.ASSISTANT
                )
                messages.append(ChatMessage(role=role, content=msg["content"]))

            # Add the current user question
            messages.append(ChatMessage(role=MessageRole.USER, content=query))

            # Call the LLM with full context
            response = self.llm.chat(messages)

            return type("Resp", (), {"response": str(response)})

    return SimpleChatEngine(llm)


def main() -> None:
    # Must be first Streamlit call
    st.set_page_config(page_title="COPD RAG Chatbot", page_icon="🫁")

    # ----- CUSTOM BACKGROUND COLOR -----
    page_bg = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #F5F7FA;  /* light grayish blue */
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    # --- Custom CSS to fix text visibility in dark & light mode ---
  
    st.markdown("""
    <style>

    /* Force all chat text to black (user + assistant) */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] * {
        color: #000 !important;
    }

    /* Fix markdown inside chat bubbles */
    [data-testid="stMarkdownContainer"] *,
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown li,
    .stMarkdown strong {
        color: #000 !important;
    }

    /* Fix expander text color */
    [data-testid="stExpander"] *,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #000 !important;
    }

    /* Fix text inside download button & other widgets */
    .stDownloadButton,
    .stDownloadButton * {
        color: #000 !important;
    }

    /* Keep your background stable */
    [data-testid="stAppViewContainer"] {
        background-color: #F5F7FA !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.title("🫁 Air Pollution & COPD – RAG Chatbot")

    st.write(
        "This chatbot uses an advanced Retrieval-Augmented Generation (RAG) system powered by "
        "semantic chunking, query rewriting, HyDE-based hypothetical answer expansion, and reranking. "
        "Your questions are rewritten for clarity, the most relevant parts of the research paper "
        "**'Air pollution and chronic obstructive pulmonary disease'** are retrieved, and the final "
        "answer is generated based on those grounded document sections."
    )

    # --- PDF download button ---
    try:
        with open("data_pdf/air_pollution.pdf", "rb") as f:
            st.download_button(
                label="⬇️ Download the PDF (Review Article)",
                data=f,
                file_name="air_pollution.pdf",
                mime="application/pdf",
            )
    except FileNotFoundError:
        st.warning(
            "⚠️ Could not find `data_pdf/air_pollution.pdf`. "
            "Check that the file exists in the `data_pdf` folder."
        )

    st.markdown("---")

    # --- Chat history in session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Manage input state ---
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    # If previous run requested clearing, do it *before* drawing the widget
    if st.session_state.clear_input:
        st.session_state.user_input = ""
        st.session_state.clear_input = False

    st.subheader("💬 Chat")

    # Show previous messages (assistant & user)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Custom input box placed right under messages ---
    with st.form("chat_form"):
        user_input = st.text_input(
            label="Your question",  # internal label for accessibility
            placeholder="Hello! Ask your question here...",
            key="user_input",
            label_visibility="collapsed",  # hides the label visually
        )
        send_btn = st.form_submit_button("Send")

    # Handle form submission
    if send_btn and user_input.strip():
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Lazily initialise chat engine & get answer from RAG engine
        with st.chat_message("assistant"):
            with st.spinner("Me, thinking extremely hard... 😤"):
                chat_engine = init_chat_engine()  # cached, heavy only on first run
                answer = chat_engine.chat(user_input)
                answer_text = getattr(answer, "response", str(answer))
                st.markdown(answer_text)

                # Show retrieved sources in expander
                if hasattr(answer, "source_nodes") and answer.source_nodes:
                    with st.expander("📄 Show retrieved document sections"):
                        for i, node in enumerate(answer.source_nodes, start=1):
                            file_name = node.metadata.get(
                                "file_name", "Unknown source"
                            )
                            pretty_name = DOC_TITLES.get(file_name, file_name)
                            st.markdown(f"### Source {i}: {pretty_name}")
                            st.write(node.text)
                            st.markdown("---")

        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text}
        )

        # Tell next run to clear the input, then force a rerun now
        st.session_state.clear_input = True
        st.rerun()

    # --- Footer Signature ---
    st.markdown(
        """
        <hr>
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            Built with passion by <b>Aslı Özdemir Strollo</b> ✨
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()