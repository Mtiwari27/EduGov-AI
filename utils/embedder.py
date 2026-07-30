from functools import lru_cache

from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model only once.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks):
    """
    Generate embeddings for document chunks.

    Args:
        chunks (list[dict]): List of document chunks.

    Returns:
        numpy.ndarray: Embeddings for all document chunks.
    """

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


def embed_query(query):
    """
    Generate an embedding for a user query.

    Args:
        query (str): User question.

    Returns:
        numpy.ndarray: Query embedding.
    """

    model = get_embedding_model()

    embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    return embedding