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

    context_parts = []

    if retrieved_docs:
        context_parts.append("Retrieved Knowledge:\n" + "\n\n".join(retrieved_docs))

    if gold_rate is not None:
        context_parts.append(f"Live Gold Rate (per gram): {gold_rate}")

    if context_parts:
        context = "\n\n".join(context_parts)
    else:
        context = "No relevant documents were retrieved for this question."


    system_prompt = (
        "You are a gold market assistant.\n\n"

        "You may use three sources of information:\n"
        "1. Retrieved context (if relevant)\n"
        "2. Live gold_rate (if provided)\n"
        "3. Your general knowledge\n\n"

        "Rules:\n"
        "- If relevant context is provided, prefer using it.\n"
        "- If context is irrelevant or missing, answer using your general knowledge.\n"
        "- Never refuse to answer solely because context is missing.\n\n"

        "If a live gold_rate is provided, it represents the 24K gold price per gram.\n\n"

        "Gold purity conversions:\n"
        "22K = gold_rate × 0.916\n"
        "18K = gold_rate × 0.75\n"
        "14K = gold_rate × 0.585\n\n"

        "Jewelry price formula:\n"
        "Base Gold Value = weight × purity_adjusted_rate\n"
        "Making Charges = Base Gold Value × making_charge_percentage\n"
        "Final Price = Base Gold Value + Making Charges\n\n"

        "Return the answer strictly in JSON format:\n"
        '{ "answer": string, "confidence": number between 0 and 1 }\n'
        "Do not include any text outside the JSON."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nReturn only JSON."

    model = genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        system_instruction=system_prompt,
    )

    try:
        response = model.generate_content(user_prompt)
        content = (response.text or "").strip()
    except Exception:
        traceback.print_exc()
        return {
            "answer": "LLM failed to generate a response.",
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