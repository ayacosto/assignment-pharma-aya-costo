"""Tools exposees a l'agent : list_medicaments et search_chunks.

Le reranking est une etape interne de search_chunks (l'agent ne le voit pas).
Les reglages techniques (k avant rerank, top_n apres) viennent de config.py.
"""

import json
import logging

from langchain_core.tools import tool

from config import RETRIEVAL_K
from retrieval import store
from retrieval.reranker import rerank_chunks

logger = logging.getLogger("graph")

NO_RELEVANT_CHUNKS = "Aucun chunk pertinent trouve pour cette recherche."


@tool
def list_medicaments() -> str:
    """Retourne la liste des noms de medicaments disponibles dans la base.
    A utiliser pour trouver le nom exact a passer a search_chunks."""
    names = store.list_medicaments()
    if not names:
        return "Aucun index trouve."
    return json.dumps(names)


@tool
def search_chunks(medicament: str, query: str) -> str:
    """Recherche dans la documentation d'UN seul medicament et retourne les
    passages les plus pertinents (avec leur source et leur breadcrumb).
    medicament: nom exact tel que retourne par list_medicaments.
    query: sujet recherche, une seule question courte."""
    vectorstore = store.load_index(medicament)
    if vectorstore is None:
        return f"Aucun index pour '{medicament}'. Utilise list_medicaments."

    results = vectorstore.similarity_search_with_score(query, k=RETRIEVAL_K)

    logger.info(f"SEARCH_CHUNKS medicament='{medicament}' query='{query}' k={RETRIEVAL_K}")
    chunks = []
    for rank, (doc, score) in enumerate(results, start=1):
        chunk = {
            "breadcrumb": doc.metadata.get("breadcrumb"),
            "source": doc.metadata.get("fichier_source"),
            "texte": doc.metadata.get("texte_original", doc.page_content),
        }
        logger.info(
            f"  AVANT RERANK [{rank}] vector_score={score:.4f} | breadcrumb={chunk['breadcrumb']}"
            f" | source={chunk['source']}\n{chunk['texte']}"
        )
        chunks.append(chunk)

    kept = rerank_chunks(query, chunks)
    logger.info(f"SEARCH_CHUNKS: {len(chunks)} chunks recuperes -> {len(kept)} gardes")
    for i, chunk in enumerate(kept, start=1):
        logger.info(
            f"  APRES RERANK [{i}] rerank_score={chunk.get('rerank_score')} | "
            f"{chunk['source']} > {chunk['breadcrumb']}\n{chunk['texte']}"
        )

    if not kept:
        return NO_RELEVANT_CHUNKS
    return json.dumps(kept, ensure_ascii=False, indent=2)


TOOLS = [list_medicaments, search_chunks]