from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env")


def _get_embeddings() -> Any:
    """
    Select embeddings backend.

    Set `EMBEDDINGS_PROVIDER=huggingface` to avoid OpenAI quota issues.
    """
    provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        return OpenAIEmbeddings()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv(
            "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {provider!r}")


def setup_vectorstore(rebuild: bool = False):
    """
    Load documents (PDF + Markdown) from ./data, split into chunks,
    embed, store in Chroma, and return a retriever.

    - If `./data/chroma` already exists and `rebuild=False`, we will LOAD the
      existing index and avoid re-embedding on every restart.
    """
    persist_dir = Path("./data/chroma")
    embeddings = _get_embeddings()

    # Fast path: load existing persisted DB (prevents re-embedding on every run)
    if persist_dir.exists() and not rebuild:
        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    # 1. Load all documents from ./data (PDFs + .md)
    pdf_loader = PyPDFDirectoryLoader("./data")
    md_loader = DirectoryLoader("./data", glob="*.md", loader_cls=TextLoader)
    documents = pdf_loader.load() + md_loader.load()

    # 2. Split documents into chunks (~400 tokens, 80 overlap approximated by characters)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )
    splits = text_splitter.split_documents(documents)

    # 3. Store in Chroma (persist to ./data/chroma)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    # 4. Return a retriever interface
    return vectorstore.as_retriever(search_kwargs={"k": 4})

