from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents):
    """
    Split documents into smaller chunks while preserving metadata.

    Args:
        documents (list[dict]): List of loaded documents.

    Returns:
        list[dict]: List of document chunks with metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for document in documents:

        split_text = splitter.split_text(document["text"])

        for chunk in split_text:

            chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "language": document["language"]
                }
            )

    return chunks