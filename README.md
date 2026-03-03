# Gold RAG

Skeleton for a Retrieval-Augmented Generation (RAG) application.

## Structure

- `app/`: application code (API, graph, nodes, vector store, prompts)
- `data/`: source documents and embeddings (to be populated)
- `requirements.txt`: Python dependencies

## Run backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run Streamlit UI

In a second terminal:

```bash
streamlit run streamlit_app.py
```

