from google import genai

from utils.config import GOOGLE_API_KEY, GEMINI_MODEL


# Create Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)


def ask_gemini(prompt: str) -> str:
    """
    Generate an answer using Gemini based on the RAG prompt.

    Args:
        prompt (str): Prompt containing the retrieved document context.

    Returns:
        str: Gemini-generated response.
    """

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text

    except Exception as error:

        return f"Error: {error}"


def ask_general_ai(question: str) -> str:
    """
    Generate a general AI response without document retrieval.

    Args:
        question (str): User's general question.

    Returns:
        str: Gemini-generated response.
    """

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question
        )

        return response.text

    except Exception as error:

        return f"Error: {error}"