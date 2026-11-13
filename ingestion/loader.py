"""Chargement des fichiers du corpus """

from pathlib import Path

EXCLUDED_FILENAMES = {"glossary.md"}


def find_corpus_files(corpus_dir: Path, extension: str = ".md") -> list[Path]:
    return [
        path
        for path in sorted(corpus_dir.rglob(f"*{extension}"))
        if path.name.lower() not in EXCLUDED_FILENAMES
    ]


def read_text(filepath: Path) -> str:
    """Lit le contenu texte d'un fichier du corpus."""
    return filepath.read_text(encoding="utf-8", errors="replace")