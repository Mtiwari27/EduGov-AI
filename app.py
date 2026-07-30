import time
import streamlit as st

from utils.config import GOOGLE_API_KEY
from utils.cache import get_vector_store
from utils.document_loader import load_documents
from utils.chunker import chunk_documents
from utils.rag import generate_answer
from utils.gemini import ask_general_ai


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EduGov AI",
    page_icon="🎓",
    layout="wide",
)

# --------------------------------------------------
# Load Resources
# --------------------------------------------------

documents = load_documents()
chunks = chunk_documents(documents)
vector_store = get_vector_store()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🎓 EduGov AI")

    if GOOGLE_API_KEY:
        st.success("✅ Gemini API Connected")
    else:
        st.error("❌ Gemini API Key Missing")

    st.markdown("---")

    st.subheader("System Statistics")

    st.metric("Documents", len(documents))
    st.metric("Chunks", len(chunks))
    st.metric("Indexed Vectors", vector_store.total_vectors())

    st.markdown("---")

    st.success("System Ready")

# --------------------------------------------------
# Government Document Assistant
# --------------------------------------------------

st.title("🎓 EduGov AI Assistant")

st.write(
    """
Ask questions related to Higher & Technical Education Department
Government Resolutions and official documents.

This assistant answers **only** from the available government documents.
"""
)

st.divider()

question = st.text_area(
    "Ask your question",
    height=100,
    placeholder="Example: What is IQAC?"
)

search = st.button(
    "🔍 Search Government Documents",
    use_container_width=True
)

if search:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        answer_start = time.time()

        with st.spinner("Searching government documents..."):

            answer, retrieved_chunks, unique_sources = generate_answer(
                question,
                vector_store
            )

        answer_time = time.time() - answer_start

        st.subheader("Government Document Answer")

        st.success(answer)

        st.caption(f"Response generated in {answer_time:.2f} seconds")

        st.divider()

        st.subheader("Retrieved Documents")

        for source in unique_sources:

            with st.expander(f"📄 {source}"):

                matching_chunks = [

                    chunk

                    for chunk in retrieved_chunks

                    if chunk["source"] == source

                ]

                for i, chunk in enumerate(matching_chunks, start=1):

                    st.markdown(f"**Chunk {i}**")

                    st.write(
                        f"Similarity Distance: {chunk['distance']:.4f}"
                    )

                    st.write(chunk["text"][:700])

                    st.markdown("---")

# --------------------------------------------------
# General AI Assistant
# --------------------------------------------------

st.divider()

st.header("🤖 AI Assistant")

st.write(
    """
Ask general questions powered by Gemini.

This assistant is **not restricted** to the government documents.
"""
)

general_question = st.text_area(
    "Ask anything",
    key="general_ai",
    height=100,
    placeholder="Example: What is Artificial Intelligence?"
)

ask_ai = st.button(
    "🤖 Ask AI",
    use_container_width=True
)

if ask_ai:

    if not general_question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            ai_answer = ask_general_ai(general_question)

        st.subheader("AI Response")

        st.success(ai_answer)