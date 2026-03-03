from typing import List, Optional, TypedDict


class RAGState:
    """Holds RAG pipeline state."""

    query: str
    retrieved_docs: List[str]
    answer: Optional[str]


class GoldAgentState(TypedDict, total=False):
    question: str
    retrieved_docs: List[str]
    gold_rate: Optional[float]
    answer: Optional[str]
    confidence: Optional[float]
    needs_live_rate: Optional[bool]
    

