import os
import streamlit as st

from utils.document_loader import load_documents
from utils.chunker import chunk_documents
from utils.embedder import embed_chunks
from utils.vector_store import VectorStore


INDEX_FILE = "embeddings/faiss_index.bin"
CHUNKS_FILE = "embeddings/chunks.pkl"


@st.cache_resource(show_spinner="Loading vector database...")
def get_vector_store():
    """
    Load an existing FAISS vector store if available.
    Otherwise, create embeddings, build the index, and save it.
    """

    vector_store = VectorStore()

    if (
        os.path.exists(INDEX_FILE)
        and
        os.path.exists(CHUNKS_FILE)
    ):

        vector_store.load()
        return vector_store

    documents = load_documents()

    chunks = chunk_documents(documents)

    embeddings = embed_chunks(chunks)

    vector_store.build_index(
        embeddings,
        chunks
    )

    vector_store.save()

    return vector_store


@st.cache_data
def get_documents():
    """
    Load and cache documents.
    """
    return load_documents()


@st.cache_data
def get_chunks(documents):
    """
    Generate and cache document chunks.
    """
    return chunk_documents(documents)


@st.cache_data
def get_embeddings(chunks):
    """
    Generate and cache embeddings.
    """
    return embed_chunks(chunks)