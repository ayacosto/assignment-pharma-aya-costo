"""Reranking des chunks recuperes par la recherche vectorielle.

Un cross-encoder (Cohere Rerank) lit la question et chaque chunk ensemble et donne
un score de pertinence entre 0 et 1. Sans cle API (ou en cas d'erreur), on garde
simplement l'ordre de la recherche vectorielle, pour que l'app marche toujours.
"""

import logging

import cohere

from config import (
    COHERE_API_KEY,
    RERANK_MIN_SCORE,
    RERANK_MODEL,
    RERANK_TOP_N,
    USE_RERANKER,
)

logger = logging.getLogger("graph")

_client = cohere.ClientV2(api_key=COHERE_API_KEY) if COHERE_API_KEY else None


def _rerank_text(chunk: dict) -> str:
    """Meme texte que pour l'embedding : breadcrumb + contenu."""
    return f"{chunk.get('breadcrumb', '')}\n\n{chunk['texte']}"


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = RERANK_TOP_N,
    min_score: float = RERANK_MIN_SCORE,
) -> list[dict]:
    """Retourne au plus top_n chunks, tries par pertinence, avec 'rerank_score'.
    Les chunks sous min_score sont ecartes. Le score de TOUS les chunks
    (gardes ou non) est logge."""
    if not chunks:
        return []

    if not USE_RERANKER:
        logger.info("RERANK: desactive (USE_RERANKER=false), ordre vectoriel conserve")
        return chunks[:top_n]

    if _client is None:
        logger.info("RERANK: pas de COHERE_API_KEY, ordre vectoriel conserve")
        return chunks[:top_n]

    try:
        response = _client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[_rerank_text(c) for c in chunks],
            top_n=len(chunks),  # on veut le score de tous les chunks pour les logs
        )
    except Exception as e:
        logger.info(f"RERANK: erreur ({e}), ordre vectoriel conserve")
        return chunks[:top_n]

    kept = []
    for result in response.results:  # tries du meilleur au moins bon
        chunk = {**chunks[result.index], "rerank_score": result.relevance_score}
        if result.relevance_score < min_score:
            status = f"ECARTE (score < {min_score})"
        elif len(kept) >= top_n:
            status = f"ECARTE (au-dela de top_n={top_n})"
        else:
            status = "GARDE"
            kept.append(chunk)
        logger.info(
            f"RERANK {status} | score={result.relevance_score:.4f} | "
            f"breadcrumb={chunk.get('breadcrumb')}"
        )
    return kept