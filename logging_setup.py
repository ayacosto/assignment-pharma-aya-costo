"""Configuration du logging 

A appeler UNE seule fois, au demarrage de l'application dans main.py.
Les autres modules font simplement : logger = logging.getLogger("graph")
"""

import logging
import sys

from config import LOG_FILE

LOGGER_NAME = "graph"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:  # deja configure : evite les logs en double
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger