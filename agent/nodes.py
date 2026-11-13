"""Noeuds du graphe : planner, retriever, answer, validator.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from agent.prompts import (
    ANSWER_REVISION_TEMPLATE,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    RETRIEVER_PLAN_TEMPLATE,
    RETRIEVER_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_USER_TEMPLATE,
)
from agent.schemas import Critique, ReasoningPlan
from agent.state import State
from config import (
    DISCLAIMER,
    MAX_REVISIONS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from retrieval.tools import TOOLS

logger = logging.getLogger("graph")

UNVERIFIED_WARNING = (
    "Warning: this answer could not be fully verified against the documentation. "
    "Points flagged by the reviewer:"
)

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0,
)
planner_llm = llm.with_structured_output(ReasoningPlan)
retriever_llm = llm.bind_tools(TOOLS)
validator_llm = llm.with_structured_output(Critique)


# --- Helpers -----------------------------------------------------------------

def extract_chunks(messages) -> list[dict]:
    """Recupere (sans doublons) les chunks renvoyes par search_chunks.
    Les reponses non JSON (ex. 'Aucun chunk pertinent') sont ignorees."""
    chunks, seen = [], set()
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != "search_chunks":
            continue
        try:
            results = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for chunk in results:
            key = (chunk.get("source"), chunk.get("breadcrumb"), chunk.get("texte"))
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)
    return chunks


def format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(no chunks found)"
    return "\n\n".join(
        f"[{i}] {c.get('source')} > {c.get('breadcrumb')}\n{c.get('texte')}"
        for i, c in enumerate(chunks, start=1)
    )


def format_critique(critique: Critique) -> str:
    """Regroupe les defauts releves par le validator en un seul texte."""
    parts = []
    if critique.unsupported_claims:
        parts.append(
            "Unsupported claims:\n" + "\n".join(f"- {c}" for c in critique.unsupported_claims)
        )
    if critique.ambiguous_claims:
        parts.append(
            "Ambiguous claims:\n" + "\n".join(f"- {c}" for c in critique.ambiguous_claims)
        )
    if critique.feedback:
        parts.append(f"Feedback:\n{critique.feedback}")
    return "\n\n".join(parts)


# --- Noeuds ------------------------------------------------------------------

def planner_node(state: State) -> dict:
    question = state["question"]
    plan = planner_llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=PLANNER_USER_TEMPLATE.format(question=question)),
        ]
    )
    logger.info(f"PLANNER key_entities={plan.key_entities}")
    logger.info(f"PLANNER research_plan={plan.research_plan}")
    logger.info(f"PLANNER search_queries={plan.search_queries}")
    logger.info(f"PLANNER reasoning_steps={plan.reasoning_steps}")
    return {
        "search_queries": plan.search_queries,
        "reasoning_steps": plan.reasoning_steps,
    }


def retriever_node(state: State) -> dict:

    messages = list(state.get("messages") or [])
    new_messages = []
    if not messages:
        queries = "\n".join(f"- {q}" for q in state["search_queries"])
        first = HumanMessage(
            content=f"Question: {state['question']}\n\n"
            + RETRIEVER_PLAN_TEMPLATE.format(queries=queries)
        )
        messages, new_messages = [first], [first]

    response = retriever_llm.invoke([SystemMessage(content=RETRIEVER_SYSTEM_PROMPT)] + messages)
    for tool_call in response.tool_calls:
        logger.info(
            f"RETRIEVER TOOL CALL: {tool_call['name']}"
            f"({json.dumps(tool_call['args'], ensure_ascii=False)})"
        )
    return {"messages": new_messages + [response]}


def answer_node(state: State) -> dict:
    
    chunks = state.get("chunks") or extract_chunks(state.get("messages", []))

    steps = "\n".join(f"- {s}" for s in state["reasoning_steps"])
    content = ANSWER_USER_TEMPLATE.format(
        question=state["question"], steps=steps, chunks=format_chunks(chunks)
    )
    if state.get("critique"):
        content += ANSWER_REVISION_TEMPLATE.format(
            previous_answer=state["answer"], critique=state["critique"]
        )

    response = llm.invoke(
        [SystemMessage(content=ANSWER_SYSTEM_PROMPT), HumanMessage(content=content)]
    )
    logger.info(f"ANSWER ({len(chunks)} chunks, revision={state.get('revisions', 0)}): {response.content}")
    return {"chunks": chunks, "answer": response.content}


def validator_node(state: State) -> dict:
    critique = validator_llm.invoke(
        [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=VALIDATOR_USER_TEMPLATE.format(
                    question=state["question"],
                    answer=state["answer"],
                    chunks=format_chunks(state["chunks"]),
                )
            ),
        ]
    )
    logger.info(
        f"VALIDATOR is_valid={critique.is_valid} "
        f"unsupported={critique.unsupported_claims} ambiguous={critique.ambiguous_claims}"
    )

    if critique.is_valid:
        return {"critique": "", "final_answer": f"{state['answer']}\n\n{DISCLAIMER}"}

    critique_text = format_critique(critique)
    revisions = state.get("revisions", 0)

    if revisions < MAX_REVISIONS:
        # Pas de final_answer : l'arete conditionnelle renvoie vers answer
        return {"critique": critique_text, "revisions": revisions + 1}

    # Limite atteinte : la reponse part avec un avertissement
    logger.info("VALIDATOR: limite de revisions atteinte, reponse envoyee avec avertissement")
    final = f"{state['answer']}\n\n{UNVERIFIED_WARNING}\n{critique_text}\n\n{DISCLAIMER}"
    return {"critique": critique_text, "final_answer": final}


def route_after_validator(state: State) -> str:
    """Fin si une reponse finale existe, sinon retour vers answer."""
    return END if state.get("final_answer") else "answer"