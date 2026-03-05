import os
import requests

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def web_search(query: str) -> str:
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {"q": query}

    res = requests.post(url, json=payload, headers=headers)
    data = res.json()

    snippets = []

    for r in data.get("organic", [])[:3]:
        snippets.append(r.get("snippet", ""))

    return "\n".join(snippets)