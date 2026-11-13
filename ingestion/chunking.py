"""Chunking hierarchique d'un document Markdown 

Etapes :
1. Decoupe par titres (h1 a h6) avec MarkdownHeaderTextSplitter.
2. Comptage des tokens (tiktoken) de chaque section.
3. Si une section depasse MAX_CHUNK_TOKENS : split recursif avec overlap.

"""

import re
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL
from ingestion.settings import (
    CHUNK_OVERLAP_RATIO,
    HEADER_KEYS,
    HEADERS_TO_SPLIT_ON,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_CHARS,
)


def _load_encoding() -> tiktoken.Encoding:
    """Encodage tiktoken du modele d'embedding """
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


_ENCODING = _load_encoding()

_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)

_SIZE_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=_ENCODING.name,
    chunk_size=MAX_CHUNK_TOKENS,
    chunk_overlap=int(MAX_CHUNK_TOKENS * CHUNK_OVERLAP_RATIO),
    separators=["\n\n", "\n", ". ", " ", ""],
)


def count_tokens(text: str) -> int:
    """Nombre de tokens du texte pour le modele d'embedding."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def extract_medicament(filename: str, file_text: str) -> str:
    """Deduit le nom du medicament depuis le titre H1 du document.
    '"""
    match = re.search(r"^#\s+(.+)$", file_text, flags=re.MULTILINE)
    if match:
        title = match.group(1).strip()
        # Retire un suffixe entre parentheses en fin de titre
        title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
        if title:
            return title.lower()
    return Path(filename).stem.lower()


def build_breadcrumb(metadata: dict) -> str:
    """Construit 'h1 > h2 > h3 ...' a partir des metadata du header splitter."""
    return " > ".join(metadata[key] for key in HEADER_KEYS if key in metadata)


def deepest_title(metadata: dict) -> str:
    """Titre le plus profond present dans la hierarchie."""
    for key in reversed(HEADER_KEYS):
        if key in metadata:
            return metadata[key]
    return ""


def split_large_section(text: str) -> list[str]:
    """Retourne [text] si la section tient dans la limite, sinon la liste
    des sous-textes (avec overlap)."""
    if count_tokens(text) <= MAX_CHUNK_TOKENS:
        return [text]
    return _SIZE_SPLITTER.split_text(text)


def chunk_document(filename: str, text: str) -> list[Document]:
    """Transforme le texte d'un fichier en chunks avec metadata complete."""
    medicament = extract_medicament(filename, text)
    chunks = []

    for section in _HEADER_SPLITTER.split_text(text):
        if len(section.page_content.strip()) < MIN_CHUNK_CHARS:
            continue  # section vide ou quasi vide

        breadcrumb = build_breadcrumb(section.metadata)
        title = deepest_title(section.metadata)

        parts = split_large_section(section.page_content)
        for i, part in enumerate(parts, start=1):
            chunks.append(
                Document(
                    page_content=part,
                    metadata={
                        "medicament": medicament,
                        "fichier_source": filename,
                        "breadcrumb": breadcrumb,
                        "titre": title,
                        "part_index": f"{i}/{len(parts)}",
                        "nb_tokens": count_tokens(part),
                    },
                )
            )

    return chunks