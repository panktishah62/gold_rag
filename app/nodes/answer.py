from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
import traceback


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=BASE_DIR / ".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

GEMINI_MODEL_NAME = "gemini-2.5-flash"


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from model output."""
    s = (text or "").strip()
    if s.startswith("```"):
        # Strip fenced blocks like ```json ... ```
        s = s.strip()
        # remove leading/backticks lines
        s = s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def generate_answer(
    question: str,
    retrieved_docs: List[str],
    gold_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate an answer using:
    - Retrieved RAG documents (if any)
    - Live gold_rate (if provided)

    Returns strict JSON:
    { "answer": str, "confidence": float }
    """

    # If absolutely no context at all, refuse
    if not retrieved_docs and gold_rate is None:
        return {
            "answer": "I do not have enough context to answer this question reliably.",
            "confidence": 0.0,
        }

    # Build context block dynamically
    context_parts = []

    if retrieved_docs:
        context_parts.append("Retrieved Knowledge:\n" + "\n\n".join(retrieved_docs))

    if gold_rate is not None:
        context_parts.append(f"Live Gold Rate (per gram): {gold_rate}")

    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a gold market assistant.\n"
        "You must answer strictly using ONLY the provided context.\n"
        "If live gold rate is provided, treat it as factual numeric data.\n"
        "If retrieved knowledge is provided, use it for rules and formulas.\n\n"
        "You must respond in strict JSON format:\n"
        '{ "answer": string, "confidence": number between 0 and 1 }\n'
        "Do not include any text outside the JSON."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Return only JSON."
    )

    model = genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        system_instruction=system_prompt,
    )

    try:
        response = model.generate_content(user_prompt)
        content = (response.text or "").strip()
    except Exception:
        print(f"Gemini generate_answer() failed (model={GEMINI_MODEL_NAME!r}):")
        traceback.print_exc()

        # Fallback
        fallback_context = context[:1500]
        return {
            "answer": (
                "LLM is unavailable right now. Available context:\n\n"
                f"{fallback_context}"
            ),
            "confidence": 0.2,
        }

    try:
        parsed = json.loads(_extract_json(content))
        answer = str(parsed.get("answer", "")).strip()
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        answer = content or "Unable to generate structured answer."
        confidence = 0.0

    return {"answer": answer, "confidence": confidence}
    """
    Use a Gemini chat model to generate an answer and a confidence score
    between 0 and 1, returned as structured JSON.
    """
    # If there is no supporting context, refuse to answer.
    if not retrieved_docs:
        return {
            "answer": "I do not have enough context to answer this question reliably.",
            "confidence": 0.0,
        }

    context = "\n\n".join(retrieved_docs)

    system_prompt = (
        "You are a question-answering system over provided context.\n"
        "You must respond in strict JSON with two fields:\n"
        '{ \"answer\": string, \"confidence\": number between 0 and 1 }.\n'
        "Base confidence on how strongly the context supports the answer."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Live gold_rate (may be null): {gold_rate}\n\n"
        f"Question:\n{question}\n\n"
        "Return only JSON, no extra text."
    )

    model = genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        system_instruction=system_prompt,
    )

    try:
        response = model.generate_content(user_prompt)
        content = (response.text or "").strip()
    except Exception:
        # If the LLM call fails, return an extractive fallback.
        print(f"Gemini generate_answer() failed (model={GEMINI_MODEL_NAME!r}):")
        traceback.print_exc()
        excerpt = "\n\n---\n\n".join(retrieved_docs)[:1500]
        return {
            "answer": (
                "LLM is unavailable right now. Here are the top retrieved excerpts:\n\n"
                f"{excerpt}"
            ),
            "confidence": 0.2,
        }

    try:
        parsed = json.loads(_extract_json(content))
        answer = str(parsed.get("answer", ""))
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        answer = content or "Unable to generate structured answer."
        confidence = 0.0

    return {"answer": answer, "confidence": confidence}


