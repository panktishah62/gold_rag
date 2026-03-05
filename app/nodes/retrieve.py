from typing import Any, List

from ..state import GoldAgentState


SIMILARITY_THRESHOLD = 0.5   # lower = more strict filtering
TOP_K = 4


def retrieve_node(state: GoldAgentState, retriever: Any) -> GoldAgentState:
    """
    LangGraph node that:
    - retrieves documents with similarity scores
    - filters out irrelevant documents
    - stores relevant page_content into state["retrieved_docs"]
    """

    question = state.get("question", "")

    relevant_docs: List[str] = []

    try:
        # Access underlying vectorstore if available (Chroma)
        vectorstore = getattr(retriever, "vectorstore", None)

        if vectorstore and hasattr(vectorstore, "similarity_search_with_score"):
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=TOP_K)

            for doc, score in docs_with_scores:
                # Chroma returns distance (lower = more similar)
                if score < SIMILARITY_THRESHOLD:
                    relevant_docs.append(doc.page_content)

        else:
            # fallback to normal retriever
            docs = retriever.invoke(question) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(question)

            relevant_docs = [doc.page_content for doc in docs[:TOP_K]]

    except Exception as e:
        print("Retrieval error:", e)
        relevant_docs = []

    # store results in state
    state["retrieved_docs"] = relevant_docs
    state["no_docs"] = len(relevant_docs) == 0

    print("retrieved_docs:", relevant_docs)

    return state