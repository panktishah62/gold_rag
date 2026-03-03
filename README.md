🏆 Gold Market RAG Agent

An AI-powered Gold Market Assistant that combines:

- 📚 Retrieval-Augmented Generation (RAG)
- 🔧 Live tool integration (GoldAPI)
- 🧠 Deterministic orchestration (LangGraph)
- ⚡ FastAPI backend
- 💬 Streamlit UI

The system answers questions about:
- Gold pricing rules
- Manufacturing formulas
- Purity calculations
- Live gold rates (INR)
- Real-time price computations

---

## 🧠 Architecture Overview

                            ┌────────────────────┐
                            │    Streamlit UI     │
                            │  (Client Interface) │
                            └──────────┬─────────┘
                                       │ HTTP POST /chat
                                       ▼
                            ┌────────────────────┐
                            │      FastAPI       │
                            │   (API Gateway)    │
                            └──────────┬─────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │    LangGraph Agent     │
                          │         (Router) │
                          └──────────┬─────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
         needs_live_rate = True            needs_live_rate = False
                    │                                 │
                    ▼                                 ▼
        ┌─────────────────────┐           ┌─────────────────────┐
        │   Tool Node         │           │   Retrieve Node     │
        │  (GoldAPI Call)     │           │   (Chroma VectorDB) │
        └──────────┬──────────┘           └──────────┬──────────┘
                   │                                  │
                   ▼                                  ▼
         ┌───────────────────┐             ┌────────────────────┐
         │  Live Gold Rate   │             │  Retrieved Chunks  │
         │  (Structured Data)│             │  (Embedded Docs)   │
         └──────────┬────────┘             └──────────┬─────────┘
                    │                                  │
                    └──────────────┬───────────────────┘
                                   ▼
                        ┌────────────────────┐
                        │  Context Builder   │
                        │ (Merge Tool + RAG) │
                        └──────────┬─────────┘
                                   ▼
                        ┌────────────────────┐
                        │ Gemini 2.5 Flash   │
                        │  (Answer Synthesis)│
                        └──────────┬─────────┘
                                   ▼
                        ┌────────────────────┐
                        │ JSON Parser +      │
                        │ Confidence Scoring │
                        └──────────┬─────────┘
                                   ▼
                            ┌────────────────┐
                            │ Final Response │
                            │  {answer, conf}│
                            └────────────────┘

🧪 Evaluation Strategy

Tested for:

- ✅ Pure RAG queries
- ✅ Pure live tool queries
- ✅ Hybrid RAG + tool queries
- ✅ Out-of-scope refusal
- ✅ Hallucination prevention

Structured JSON output includes:
- `answer`
- `confidence` (0–1 heuristic)

---

 🔐 Important Design Decisions

- No hardcoded gold price
- No LLM-based routing
- No hallucinated external knowledge
- Strict JSON output enforcement
- Live API failure surfaces clearly (no silent fallback)

📈 Future Improvements

- Deterministic price calculation (remove LLM math)
- Retrieval similarity thresholding
- Gold rate caching (TTL-based)
- Streaming token responses
- Structured tool invocation with schema validation
- Observability (latency + routing metrics)
- Multi-tool expansion (USD conversion, historical analysis)

---

 🎯 Summary

This project demonstrates:

- Hybrid RAG + Tool orchestration
- Clean agent architecture
- Real-time financial data integration
- Production-aware design choices
- Controlled hallucination behavior

Built as an interview MVP with scalability in mind.
