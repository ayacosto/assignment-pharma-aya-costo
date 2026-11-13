"""Sorties structurees (Pydantic) des appels LLM 
"""

from pydantic import BaseModel


class ReasoningPlan(BaseModel):
    """Sortie du noeud planner."""

    key_entities: list[str]
    research_plan: str
    search_queries: list[str]
    reasoning_steps: list[str]


class Critique(BaseModel):
    """Sortie du noeud validator."""

    is_valid: bool
    unsupported_claims: list[str]
    ambiguous_claims: list[str]
    feedback: str  # texte renvoye a answer pour corriger ("" si is_valid)