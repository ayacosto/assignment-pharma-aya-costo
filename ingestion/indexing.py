"""Embedding par batch et sauvegarde d'un index FAISS par medicament.
"""

import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from ingestion.settings import (
    BATCH_PAUSE_SECONDS,
    EMBEDDING_BATCH_SIZE,
    MAX_RETRIES,
    RETRY_BASE_WAIT_SECONDS,
)


def medicament_index_dir(base_dir: Path, medicament: str) -> Path:
    """Dossier de l'index d'un medicament."""
    safe_name = re.sub(r"[^a-z0-9_-]+", "_", medicament.lower()).strip("_")
    return Path(base_dir) / (safe_name or "unknown")


def group_by_medicament(docs: list[Document]) -> dict[str, list[Document]]:
    """Regroupe les chunks par medicament (un index FAISS par medicament)."""
    grouped: dict[str, list[Document]] = {}
    for doc in docs:
        grouped.setdefault(doc.metadata["medicament"], []).append(doc)
    return grouped


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def embed_batch_with_retry(
    vectorstore: FAISS | None,
    embeddings: OpenAIEmbeddings,
    texts: list[str],
    metadatas: list[dict],
    label: str,
) -> FAISS:
    """Embedde un batch et l'ajoute au vectorstore"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if vectorstore is None:
                return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,)
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
            return vectorstore
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Echec definitif sur {label}") from e
            wait = RETRY_BASE_WAIT_SECONDS * attempt
            print(f"    Erreur ({e}). Nouvelle tentative dans {wait}s...")
            time.sleep(wait)


def build_index_for_medicament(
    medicament: str,
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    base_dir: Path,
) -> Path:
    """Embedde tous les chunks d'un medicament par batch et sauvegarde l'index."""
    texts = [f"{d.metadata['breadcrumb']}\n\n{d.page_content}" for d in docs]
    metadatas = [{**d.metadata, "texte_original": d.page_content} for d in docs]

    n_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    vectorstore = None
    for batch_num, start in enumerate(range(0, len(texts), EMBEDDING_BATCH_SIZE), start=1):
        end = start + EMBEDDING_BATCH_SIZE
        print(f"  Batch {batch_num}/{n_batches} ({len(texts[start:end])} chunks)...")
        vectorstore = embed_batch_with_retry(
            vectorstore,
            embeddings,
            texts[start:end],
            metadatas[start:end],
            label=f"batch {batch_num} ({medicament})",
        )
        time.sleep(BATCH_PAUSE_SECONDS)

    index_dir = medicament_index_dir(base_dir, medicament)
    vectorstore.save_local(str(index_dir))
    return index_dir


def build_all_indexes(docs: list[Document], base_dir: Path) -> int:
    """Construit et sauvegarde un index par medicament."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    embeddings = create_embeddings()

    grouped = group_by_medicament(docs)
    for medicament, med_docs in grouped.items():
        print(f"\nMedicament: {medicament} ({len(med_docs)} chunks)")
        index_dir = build_index_for_medicament(medicament, med_docs, embeddings, base_dir)
        print(f"  Index sauvegarde dans: {index_dir}/")

    return len(grouped)