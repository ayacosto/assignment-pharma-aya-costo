"""Etat partage du graphe : Chaque noeud renvoie uniquement les cles qu'il met a jour.
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Entree
    question: str

    # Sortie du planner
    search_queries: list[str]
    reasoning_steps: list[str]

    # Boucle retriever (agent <-> tools)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Chunks gardes (extraits des resultats de search_chunks, lus par answer et validator)
    chunks: list[dict]

    # Generation et validation
    answer: str
    critique: str       # texte du validator expliquant ce qui ne va pas ("" si valide)
    revisions: int      # nb de retours du validator vers answer

    # Sortie finale (reponse + disclaimer)
    final_answer: str