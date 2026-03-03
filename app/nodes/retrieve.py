from typing import Any, List

from ..state import GoldAgentState


def retrieve_node(state: GoldAgentState, retriever: Any) -> GoldAgentState:
    """
    LangGraph node that:
    - takes GoldAgentState
    - uses a retriever to fetch top 4 relevant docs
    - stores page_content into state["retrieved_docs"]
    - returns the updated state
    """
    question = state.get("question", "")
    # LangChain retriever API changed over time.
    # Prefer `.invoke(query)`; fall back to legacy methods if needed.
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    elif hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    else:
        docs = retriever._get_relevant_documents(question)  # type: ignore[attr-defined]

    if not docs:
        # No documents found: mark flag so the graph can skip tools
        state["retrieved_docs"] = []
        state["no_docs"] = True
        return state

    top_docs = list(docs)[:4]
    state["retrieved_docs"] = [doc.page_content for doc in top_docs]
    state["no_docs"] = False

    # Debug: print retrieved docs during testing
    print("retrieved_docs:", state["retrieved_docs"])

    return state


