"""Point d'entree de l'ingestion
"""

import argparse
from pathlib import Path

from config import DEFAULT_INDEX_DIR
from ingestion.chunking import chunk_document
from ingestion.indexing import build_all_indexes
from ingestion.loader import find_corpus_files, read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--extension", type=str, default=".md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = find_corpus_files(args.corpus_dir, args.extension)
    if not files:
        print(f"Aucun fichier '{args.extension}' trouve dans {args.corpus_dir}")
        return

    print(f"{len(files)} fichiers trouves. Chunking en cours...")
    all_chunks = []
    for filepath in files:
        chunks = chunk_document(filepath.name, read_text(filepath))
        all_chunks.extend(chunks)
        print(f"  {filepath.name}: {len(chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks a embedder.")
    n_indexes = build_all_indexes(all_chunks, args.index_dir)
    print(f"\nTermine. {n_indexes} index FAISS crees sous {args.index_dir}/ (un par medicament).")


if __name__ == "__main__":
    main()