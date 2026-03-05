from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
import google.generativeai as genai

from ..state import GoldAgentState

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=BASE_DIR / ".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

GEMINI_MODEL_NAME = "gemini-2.5-flash"


def decision_node(state: GoldAgentState) -> GoldAgentState:
    """
    Decide whether the question needs a *live* gold rate.

    Hybrid routing:
    - First: deterministic keyword-based detection.
    - Second: LLM fallback only for ambiguous gold/rate questions.
    """
    question = cast(str, state.get("question", ""))
    q = question.lower()

    keywords = ["today", "current", "live", "now", "latest"]
    needs_live = any(word in q for word in keywords) and (
        "gold" in q or "rate" in q or "price" in q
    )

    if needs_live:
        state["needs_live_rate"] = True
        return state

    is_gold_related = any(w in q for w in ("gold", "rate", "price"))
    if is_gold_related:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            prompt = (
                "You are a classifier.\n"
                "Does this question require the *current live gold rate* to answer?\n"
                "Respond with exactly one word: YES or NO.\n\n"
                f"Question: {question}"
            )
            resp = model.generate_content(prompt)
            content = (getattr(resp, "text", "") or "").strip().upper()
            needs_live = content.startswith("YES")
        except Exception:
            needs_live = False

    state["needs_live_rate"] = needs_live
    return state


