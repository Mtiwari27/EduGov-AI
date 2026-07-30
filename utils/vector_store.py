import os
import pickle

import faiss
import numpy as np


class VectorStore:
    """
    FAISS vector store with persistent storage.
    """

    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, embeddings, chunks):
        """
        Build a FAISS index from document embeddings.

        Args:
            embeddings (numpy.ndarray): Document embeddings.
            chunks (list[dict]): Document chunks with metadata.
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        self.chunks = chunks

    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar document chunks.

        Args:
            query_embedding (numpy.ndarray): Embedded user query.
            top_k (int): Number of nearest neighbors to retrieve.

        Returns:
            list[dict]: Retrieved chunks with similarity distance.
        """

        if self.index is None:
            raise ValueError("FAISS index not loaded.")

        query_embedding = np.asarray(
            [query_embedding],
            dtype=np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()

            chunk["distance"] = float(distance)

            results.append(chunk)

        return results

    def save(self, folder="embeddings"):
        """
        Save the FAISS index and document chunks.
        """

        os.makedirs(folder, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(folder, "faiss_index.bin")
        )

        with open(
            os.path.join(folder, "chunks.pkl"),
            "wb"
        ) as f:

            pickle.dump(self.chunks, f)

    def load(self, folder="embeddings"):
        """
        Load the FAISS index and document chunks.

        Returns:
            bool: True if the index was loaded successfully,
                  otherwise False.
        """

        index_path = os.path.join(
            folder,
            "faiss_index.bin"
        )

        chunk_path = os.path.join(
            folder,
            "chunks.pkl"
        )

        if (
            not os.path.exists(index_path)
            or
            not os.path.exists(chunk_path)
        ):
            return False

        self.index = faiss.read_index(index_path)

        with open(chunk_path, "rb") as f:
            self.chunks = pickle.load(f)

        return True

    def total_vectors(self):
        """
        Return the total number of indexed vectors.
        """

        if self.index is None:
            return 0

        return self.index.ntotal