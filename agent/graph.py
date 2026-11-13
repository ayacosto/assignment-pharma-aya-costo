"""Assemblage du graphe LangGraph.

    planner -> retriever <-> tools
                  |
                  v
               answer <-> validator -> fin

"""

import logging

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import (
    answer_node,
    planner_node,
    retriever_node,
    route_after_validator,
    validator_node,
)
from agent.state import State
from retrieval.tools import TOOLS

logger = logging.getLogger("graph")

RECURSION_LIMIT = 60 


def build_graph():
    graph = StateGraph(State)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("answer", answer_node)
    graph.add_node("validator", validator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_conditional_edges(
        "retriever", tools_condition, {"tools": "tools", END: "answer"}
    )
    graph.add_edge("tools", "retriever")
    graph.add_edge("answer", "validator")
    graph.add_conditional_edges(
        "validator", route_after_validator, {"answer": "answer", END: END}
    )
    return graph.compile()


app = build_graph()


def ask(question: str) -> dict:
    """Execute tout le pipeline. Retourne la reponse finale (avec disclaimer)
    et les etapes intermediaires (pour l'onglet 'etapes' du frontend)."""
    logger.info("=" * 80)
    logger.info(f"QUESTION: {question}")

    final_state = app.invoke(
        {"question": question, "revisions": 0},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    answer = final_state["final_answer"]
    logger.info(f"FINAL_ANSWER: {answer}")
    return {
        "answer": answer,
        "details": {
            "search_queries": final_state.get("search_queries", []),
            "reasoning_steps": final_state.get("reasoning_steps", []),
            "chunks": final_state.get("chunks", []),
            "revisions": final_state.get("revisions", 0),
            "critique": final_state.get("critique", ""),
        },
    }