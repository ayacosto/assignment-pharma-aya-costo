"""Acces aux index FAISS : liste des medicaments et chargement avec cache."""

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from config import DEFAULT_INDEX_DIR
from ingestion.indexing import create_embeddings, medicament_index_dir


@lru_cache(maxsize=1)
def _embeddings():
    return create_embeddings()


def list_medicaments() -> list[str]:
    """Noms (noms de dossiers) des medicaments disponibles, tries."""
    if not DEFAULT_INDEX_DIR.exists():
        return []
    return sorted(p.name for p in DEFAULT_INDEX_DIR.iterdir() if p.is_dir())


@lru_cache(maxsize=None)
def load_index(medicament: str) -> FAISS | None:
    """Charge l'index d'un medicament (une seule fois grace au cache)."""
    index_dir = medicament_index_dir(DEFAULT_INDEX_DIR, medicament)
    if not index_dir.exists():
        return None
    return FAISS.load_local(
        str(index_dir),
        _embeddings(),
        allow_dangerous_deserialization=True,  
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, 
    )