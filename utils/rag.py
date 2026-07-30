from collections import OrderedDict

from utils.config import TOP_K
from utils.embedder import embed_query
from utils.gemini import ask_gemini


def build_prompt(question, context):
    """
    Build the prompt sent to Gemini.

    Args:
        question (str): User's question.
        context (str): Retrieved document context.

    Returns:
        str: Prompt for Gemini.
    """

    return f"""
You are EduGov AI, an AI assistant for the Higher & Technical Education Department, Government of Maharashtra.

You MUST answer ONLY using the provided government document context.

Rules:
1. Use ONLY the information present in the context.
2. Do NOT use your own knowledge.
3. If the answer is not found in the context, reply exactly:
"I couldn't find this information in the provided government documents."
4. If available, include:
   • Government Resolution Number
   • Date
   • Department
   • Eligibility
   • Conditions
5. Keep the answer concise, accurate and well-structured.

-----------------------------
CONTEXT
-----------------------------
{context}

-----------------------------
QUESTION
-----------------------------
{question}

-----------------------------
ANSWER
-----------------------------
"""


def generate_answer(question, vector_store):
    """
    Execute the complete Retrieval-Augmented Generation (RAG) pipeline.

    Args:
        question (str): User's question.
        vector_store (VectorStore): FAISS vector store.

    Returns:
        tuple:
            - answer (str)
            - retrieved_chunks (list[dict])
            - unique_sources (list[str])
    """

    # Step 1: Generate query embedding
    query_embedding = embed_query(question)

    # Step 2: Retrieve relevant document chunks
    retrieved_chunks = vector_store.search(
        query_embedding,
        top_k=TOP_K
    )

    # Step 3: Build context
    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    # Step 4: Create prompt
    prompt = build_prompt(
        question,
        context
    )

    # Step 5: Generate answer
    answer = ask_gemini(prompt)

    # Step 6: Collect unique document sources
    unique_sources = list(
        OrderedDict.fromkeys(
            chunk["source"]
            for chunk in retrieved_chunks
        )
    )

    return answer, retrieved_chunks, unique_sources