# Customer Service Chatbot

A full-stack, LLM-powered customer service chatbot built as a hands-on learning project for **LangChain**. It uses **Google Gemini** as the underlying model, **FastAPI** as the web backend, and a minimal HTML/JS frontend.

This project was built incrementally, component by component, to gain practical familiarity with all six core LangChain building blocks: **Models, Prompts, Chains, Memory, Indexes/Retrievers, and Agents/Tools**.

## Status

🚧 **In progress** — core chatbot is functional; RAG (retrieval) is being added; Agents/Tools are pending.

| Component | Status | Notes |
|---|---|---|
| Models | ✅ Done | `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) |
| Prompts | ✅ Done | `ChatPromptTemplate` with a customer-service system prompt |
| Chains | ✅ Done | LCEL (`prompt \| llm \| StrOutputParser()`) |
| Memory | ✅ Done | Per-session history via `MessagesPlaceholder`, `HumanMessage`/`AIMessage` |
| Indexes / Retrievers (RAG) | 🚧 In progress | Document loading & chunking started (`TextLoader`, `RecursiveCharacterTextSplitter`); embeddings + vector store next |
| Agents / Tools | ⏳ Pending | Planned: tool-based actions (e.g. order status lookup) |

## Tech Stack

- **LangChain** — orchestration (prompts, chains, memory, retrieval)
- **Google Gemini** (`gemini-2.5-flash`) — LLM, via `langchain-google-genai`
- **FastAPI** — web backend, with a `/chat` endpoint and CORS enabled
- **FAISS** (`faiss-cpu`) — local vector store for RAG
- **HTML/JS** — minimal browser frontend (no framework)
- **Python** (virtual environment, WSL)

## Project Structure

```
.
├── main.py          # Core chatbot logic: build_chain(), get_response()
├── app.py           # FastAPI backend, /chat endpoint, session management
├── index.html        # Minimal browser chat UI
├── data/
│   └── faq.txt        # Knowledge base source for RAG (FAQs/policies)
├── .env               # GOOGLE_API_KEY (not committed)
└── README.md
```

## Setup

1. **Clone and create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install langchain langchain-google-genai python-dotenv fastapi uvicorn langchain-community faiss-cpu langchain-text-splitters
   ```

3. **Add your Gemini API key** in a `.env` file at the project root:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
   Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

## Running

**Backend (FastAPI):**
```bash
uvicorn app:app --reload
```
- API docs: `http://127.0.0.1:8000/docs`
- Chat endpoint: `POST /chat` — body: `{"session_id": "...", "message": "..."}`

**Frontend:**
Open `index.html` directly in a browser. It talks to the FastAPI backend at `http://127.0.0.1:8000`.

**CLI (for quick testing without the web layer):**
```bash
python main.py
```

## Known Limitations

- Gemini's free tier has a low daily request quota (`gemini-2.5-flash`); expect occasional `429 RESOURCE_EXHAUSTED` errors during heavy testing.
- Conversation history is stored in-memory (a `sessions` dict) — it resets on server restart. No database persistence yet.
- No authentication on the `/chat` endpoint.
- CORS is currently wide open (`allow_origins=["*"]`) for local development — should be locked down before any real deployment.
