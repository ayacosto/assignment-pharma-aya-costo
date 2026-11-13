"""Configuration : lecture de .env """

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- API OpenAI (embedding + chat) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Stockage des index FAISS ---
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss_index"

# --- Recherche et reranking ---
RETRIEVAL_K = 5          # chunks recuperes par la recherche vectorielle 
COHERE_API_KEY = os.getenv("COHERE_API_KEY") 
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v3.5")
RERANK_TOP_N = 3         # chunks gardes apres reranking
RERANK_MIN_SCORE = 0.3    # score minimal à calibrer
USE_RERANKER = True

# --- Validation de la reponse ---
MAX_REVISIONS = 2         
# --- Sortie ---
DISCLAIMER = (
    "This information is for educational purposes only and is not medical advice. "
    "Consult a doctor or pharmacist before making any decision about your treatment."
)

# --- Logs ---
LOG_FILE = PROJECT_ROOT / "graph.log"