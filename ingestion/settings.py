"""Constantes propres a l'ingestion"""

# --- Chunking ---
MAX_CHUNK_TOKENS = 500              
CHUNK_OVERLAP_RATIO = 0.15
MIN_CHUNK_CHARS = 5            

MAX_HEADER_LEVEL = 6            
HEADER_KEYS = [f"h{level}" for level in range(1, MAX_HEADER_LEVEL + 1)]
HEADERS_TO_SPLIT_ON = [
    ("#" * level, f"h{level}") for level in range(1, MAX_HEADER_LEVEL + 1)
]

# --- Embedding ---
EMBEDDING_BATCH_SIZE = 50      
BATCH_PAUSE_SECONDS = 2        
MAX_RETRIES = 5
RETRY_BASE_WAIT_SECONDS = 10    