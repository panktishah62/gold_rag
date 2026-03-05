from typing import Any, cast
from functools import partial

from langgraph.graph import StateGraph

from .state import GoldAgentState
from .vectorstore import setup_vectorstore
from .nodes.retrieve import retrieve_node
from .nodes.decision import decision_node
from .nodes.tools import get_live_gold_rate
from .nodes.answer import generate_answer


def tool_node(state: GoldAgentState) -> GoldAgentState:
    """Fetch live gold rate."""
    rate = get_live_gold_rate()
    state["gold_rate"] = rate
    return state


def answer_node(state: GoldAgentState) -> GoldAgentState:
    """Generate final answer."""
    question = cast(str, state.get("question", ""))
    retrieved_docs = cast(list[str], state.get("retrieved_docs", [])) or []
    gold_rate = state.get("gold_rate")

    result = generate_answer(
        question=question,
        retrieved_docs=retrieved_docs,
        gold_rate=gold_rate,
    )

    state["answer"] = result.get("answer")

    try:
        state["confidence"] = float(result.get("confidence") or 0.0)
    except Exception:
        state["confidence"] = 0.0

    return state


def route_from_decision(state: GoldAgentState) -> str:
    """Route based on live rate need."""
    needs_live = bool(state.get("needs_live_rate"))
    return "tool_branch" if needs_live else "retrieve_branch"


def build_graph() -> Any:
    """Build LangGraph workflow."""

    retriever = setup_vectorstore()
    retrieve_node_bound = partial(retrieve_node, retriever=retriever)

    graph = StateGraph(GoldAgentState)

    graph.add_node("decision_node", decision_node)
    graph.add_node("retrieve_node", retrieve_node_bound)
    graph.add_node("tool_node", tool_node)
    graph.add_node("answer_node", answer_node)

    # Entry starts at decision (does it need live rate?)
    graph.set_entry_point("decision_node")

    # If live rate is needed, fetch it first; then retrieve docs; then answer.
    # Otherwise, retrieve docs; then answer.
    graph.add_conditional_edges(
        "decision_node",
        route_from_decision,
        {
            "tool_branch": "tool_node",
            "retrieve_branch": "retrieve_node",
        },
    )

    graph.add_edge("tool_node", "retrieve_node")
    graph.add_edge("retrieve_node", "answer_node")

    graph.set_finish_point("answer_node")

    return graph.compile()