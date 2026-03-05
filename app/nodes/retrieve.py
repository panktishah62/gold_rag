from typing import Any, List

from ..state import GoldAgentState

def retrieve_node(state: GoldAgentState, retriever: Any) -> GoldAgentState:

    question = state.get("question", "")

    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    elif hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    else:
        docs = retriever._get_relevant_documents(question)

    top_docs = list(docs)[:4]

    # Extract text
    state["retrieved_docs"] = [doc.page_content for doc in top_docs]

    return state