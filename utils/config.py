import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================
# Gemini Configuration
# ============================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini model
GEMINI_MODEL = "gemini-flash-latest"

# ============================
# Embedding Configuration
# ============================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================
# RAG Configuration
# ============================

# Number of chunks to retrieve
TOP_K = 5

# Chunk size
CHUNK_SIZE = 500

# Overlap between chunks
CHUNK_OVERLAP = 100