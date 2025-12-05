# app.py

import streamlit as st

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
}


@st.cache_resource
def init_chat_engine() -> object:
    """Load LLM, embeddings and RAG engine only once."""
    llm: GoogleGenAI = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()
    chat_engine = get_chat_engine(llm=llm, embed_model=embed_model)
    return chat_engine


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

    # --- Custom CSS: tighten layout & center content ---
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 850px;
            margin: auto;
        }
        h1 {
            margin-bottom: 0.5rem !important;
        }
        .stMarkdown p {
            margin-bottom: 0.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        with open("data/air_pollution.pdf", "rb") as f:
            st.download_button(
                label="⬇️ Download the PDF (Review Article)",
                data=f,
                file_name="air_pollution.pdf",
                mime="application/pdf",
            )
    except FileNotFoundError:
        st.warning("⚠️ Could not find `data/air_pollution.pdf`. Check the file name and path.")

    st.markdown("---")

    # --- Initialise chat engine ---
    chat_engine = init_chat_engine()

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

        # Get answer from RAG engine
        with st.chat_message("assistant"):
            with st.spinner("Me, thinking extremely hard... 😤"):
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