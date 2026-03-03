# from typing import Any, cast
# from functools import partial

# from langgraph.graph import StateGraph

# from .state import GoldAgentState
# from .vectorstore import setup_vectorstore
# from .nodes.retrieve import retrieve_node
# from .nodes.decision import decision_node
# from .nodes.tools import get_live_gold_rate
# from .nodes.answer import generate_answer


# def tool_node(state: GoldAgentState) -> GoldAgentState:
#     """Node that fetches a live gold rate and stores it in the state."""
#     rate = get_live_gold_rate()
#     state["gold_rate"] = rate
#     return state


# def answer_node(state: GoldAgentState) -> GoldAgentState:
#     """Node that generates an answer using the question and retrieved docs."""
#     question = cast(str, state.get("question", ""))
#     retrieved_docs = cast(list[str], state.get("retrieved_docs", []))
#     result = generate_answer(
#         question=question,
#         retrieved_docs=retrieved_docs,
#         gold_rate=cast(float | None, state.get("gold_rate")),
#     )
#     print("Question:", question)
#     print("needs_live_rate:", state.get("needs_live_rate"))
#     state["answer"] = result.get("answer")
#     try:
#         state["confidence"] = float(result.get("confidence") or 0.0)
#     except Exception:
#         state["confidence"] = 0.0
#     return state


# def route_from_retrieve(state: GoldAgentState) -> str:
#     """Routing function after retrieval.

#     If no documents were found, skip directly to answer_node.
#     Otherwise, proceed to decision_node.
#     """
#     no_docs = bool(state.get("no_docs"))
#     return "no_docs" if no_docs else "has_docs"


# def route_from_decision(state: GoldAgentState) -> str:
#     """Routing function based on whether a live rate is needed."""
#     needs_live = bool(state.get("needs_live_rate"))
#     return "tool_branch" if needs_live else "answer_branch"


# def build_graph() -> Any:
#     """Build and return a LangGraph graph for GoldAgentState."""
#     # Set up retriever once and bind it into the retrieve_node
#     retriever = setup_vectorstore()
#     retrieve_node_bound = partial(retrieve_node, retriever=retriever)

#     graph = StateGraph(GoldAgentState)

#     # Nodes
#     graph.add_node("retrieve_node", retrieve_node_bound)
#     graph.add_node("decision_node", decision_node)
#     graph.add_node("tool_node", tool_node)
#     graph.add_node("answer_node", answer_node)

#     # Flow: retrieve -> (answer if no docs) or decision -> (tool -> answer) or answer directly
#     graph.set_entry_point("retrieve_node")

#     graph.add_conditional_edges(
#         "retrieve_node",
#         route_from_retrieve,
#         {
#             "no_docs": "answer_node",
#             "has_docs": "decision_node",
#         },
#     )

#     graph.add_conditional_edges(
#         "decision_node",
#         route_from_decision,
#         {
#             "tool_branch": "tool_node",
#             "answer_branch": "answer_node",
#         },
#     )

#     graph.add_edge("tool_node", "answer_node")

#     graph.set_finish_point("answer_node")

#     return graph.compile()

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
    print("Tool executed. Gold rate:", rate)
    return state


def answer_node(state: GoldAgentState) -> GoldAgentState:
    """Generate final answer."""
    question = cast(str, state.get("question", ""))
    retrieved_docs = cast(list[str], state.get("retrieved_docs", []))
    gold_rate = state.get("gold_rate")

    result = generate_answer(
        question=question,
        retrieved_docs=retrieved_docs,
        gold_rate=gold_rate,
    )

    print("Question:", question)
    print("needs_live_rate:", state.get("needs_live_rate"))

    state["answer"] = result.get("answer")

    try:
        state["confidence"] = float(result.get("confidence") or 0.0)
    except Exception:
        state["confidence"] = 0.0

    return state


def route_from_decision(state: GoldAgentState) -> str:
    """Route based on live rate need."""
    needs_live = bool(state.get("needs_live_rate"))
    print("Routing decision:", needs_live)

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

    # ENTRY STARTS AT DECISION
    graph.set_entry_point("decision_node")

    # Decision routing
    graph.add_conditional_edges(
        "decision_node",
        route_from_decision,
        {
            "tool_branch": "tool_node",
            "retrieve_branch": "retrieve_node",
        },
    )

    # Tool → Answer
    graph.add_edge("tool_node", "answer_node")

    # Retrieve → Answer
    graph.add_edge("retrieve_node", "answer_node")

    graph.set_finish_point("answer_node")

    return graph.compile()