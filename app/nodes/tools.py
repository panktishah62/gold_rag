"""Tool definitions used by the RAG graph."""

import os
from datetime import datetime
import requests


def get_live_gold_rate() -> float:
    api_key = os.getenv("GOLD_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOLDAPI_KEY")

    symbol = "XAU"
    currency = "INR"


    url = f"https://www.goldapi.io/api/{symbol}/{currency}"

    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()
    print("GoldAPI response:", data)

    # Inspect actual field returned
    # Common fields:
    # - price
    # - price_gram_24k
    # - price_gram_22k

    if "price_gram_24k" in data:
        return float(data["price_gram_24k"])
    elif "price" in data:
        # Sometimes this is per ounce
        price_per_ounce = float(data["price"])
        return price_per_ounce / 31.1035
    else:
        raise ValueError(f"Unexpected GoldAPI schema: {data}")