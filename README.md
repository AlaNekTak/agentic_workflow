# 🤖 Agentic Workflow — Biomedical Research Assistant  
*A multi-step LangGraph agent with Gradio UI*

## 📑 Table of Contents
1. [Overview](#overview)  
2. [Quick Start](#quick-start)  
3. [Repository Layout](#repository-layout)  
4. [Agentic Workflow Diagram](#agentic-workflow-diagram)  
5. [FAQ](#faq)

---

## 🔍 Overview<a id="overview"></a>

This project shows how **agentic workflows** can combine
LangChain + LangGraph + Gradio + SQLite + Chroma + external search to:

* 🔎 search web / Semantic Scholar / arXiv and scrape content  
* 📚 query a **local literature cache** (vector RAG)  
* 🗄️ run **SQL** against a SQLite paper database and auto-explain the result  
* 🖼️ render first-page thumbnails of PDFs in an interactive gallery  

Everything is orchestrated by a LangGraph graph, wrapped in three
tools (`run_web_graph`, `run_local_rag`, `run_sql_graph`) that a
higher-level **functions agent** can call when chatting with the user.

---

## ⚡ Quick Start<a id="quick-start"></a>

```bash
# 1 – clone
git clone https://github.com/AlaNekTak/agentic_workflow.git

# 2 – create venv & install & set your api keys
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3 – run the UI
python ui.py

```

## 🗂️ Repository Layout <a id="repository-layout"></a>
| Path | Purpose |
| --- |  --- |
| **ui.py** | 📟 Launches the Gradio interface and wires the agent into it. |
| **app.py** | 🧠 Core LangGraph graphs, tools and agent definitions. |
| **search\_tools.py** | 🌐 Wraps Google SERP, Semantic Scholar and arXiv search + scraping. |
| **local\_db.py** | 📚 Creates/updates the `papers` SQLite DB and builds a Chroma vectorstore for local RAG. |
| **models.py** | 🔑 Factory for `get_chat_model()` and `get_embeddings()`.                                |
| **docs/** | Example CSV / paper metadata used to bootstrap the SQLite DB. |
| **pic.png** | Tiny logo shown in the header. |
| **eval/** | Notebook(s) and test prompts for offline evaluation. |
| **requirements.txt** | Python dependencies (LangChain, LangGraph, Gradio, pdf2image, …). |
| **README.md** | ← you’re here. |



### 🧠 app.py

| Section | Contents |
|---------|----------|
| **Imports / Globals** | Heavy use of LangGraph & LangChain (`StateGraph`, `ChatPromptTemplate`, `AgentExecutor`), plus `pdf2image`, **PyMuPDF** (`fitz`) for thumbnails, `BeautifulSoup`, `pandas`, `pydantic`.  Global singletons: `store` (CSV → SQLite → Chroma), `web_helper`, two OpenAI models (`llm_big`, `llm_fast`). |
| **Helper Functions** | `sql_df()`, small utils for safe‐formatting contexts & base-64 PNG conversion. |
| **WebInsightGraph** | Pipeline **fetch_web → scrape → pdf_thumbs → reflect**.<br>• `fetch_web` calls the combined search-tool (Google + S2 + arXiv) and returns *summary + urls + pdf_urls*.<br>• `scrape` bulk-pulls HTML (3 kB/page).<br>• `pdf_thumbs` renders first page of each PDF (pdf2image → fallback to PyMuPDF → fallback icon).<br>• `reflect` feeds the stitched CONTEXT to `llm_big`. |
| **Local-RAG Graph** | Minimal 2-node chain (`search_local`, `reflect_local`) that queries the on-disk Chroma index only. |
| **SQLInsightGraph** | 3 nodes **(+ retry loop)**: <br>1. `write_sql` – LLM writes *one plain* SQL query (markdown fences stripped).<br>2. `run_sql` – executes, catches errors, sets `failed` / `last_error`.<br>3. Conditional edge: on failure ⇒ back to `write_sql`, else `summarise` (LLM explains DataFrame). |
| **Graph Wrappers** | `_run_graph()` streams LangGraph updates → collects `thumb_buf` for the UI and pretty-prints JSON into `trace_buf`. |
| **LangChain Tools** | Three `StructuredTool`s: **run_web_graph**, **run_local_rag**, **run_sql_graph** (latter enforces plain-English input with a custom `RunSQLIn` schema). |
| **build_agent()** | Assembles the smart “biomedical research assistant”:<br>• Rich system prompt guides tool-choice.<br>• Injects user chat history via `MessagesPlaceholder("messages")`.<br>• Returns an `AgentExecutor` wired to the three tools. |
| **UI (Gradio)** | Moved to `ui.py` but driven by the agent built here. Features:<br>• Front-tab chat + dynamic PDF thumbnail gallery.<br>• Back-tab trace accordion.<br>• `dispatch()` manages chat history, passes `messages` to the agent, surfaces `thumb_buf` when available. |



### 🌐 search_tools.py

| Section | Contents |
|---------|----------|
| **Imports / Globals** | `requests`, `BeautifulSoup`, `time`, `arxiv`, `os` for API keys (`SERP_KEY`, `S2_KEY`); Pydantic `BaseModel`, LangChain `StructuredTool`; helper `_throttle()` 1 req/s. |
| **_google_serp_search()** | Uses **SerpAPI** Google endpoint → returns human-readable context blocks **plus** structured dicts ready for PaperStore. |
| **_s2_api_search()** | Calls **Semantic Scholar Graph API** (`/paper/search`) – same dual output (context + structs). |
| **_arxiv_search()** | Leverages `arxiv.Client()` → pulls title/abstract, rewrites `/abs/` links → `/pdf/` when harvesting. |
| **_pick_emoji()** | Tiny helper that tags each source with 📄 / 🔬 / 🧬 emojis used in summaries. |
| **_add_structs()** | Dedup-check against SQLite (`PaperStore._is_duplicate`) then bulk-append to PaperStore (CSV + SQLite + Chroma). |
| **fetch_all()** | Convenience method that runs **all three searches**, caches everything, and returns a stitched text summary. |
| **make_combined_search_tool()** | Factory that builds a LangChain **functions tool**. Returns `(summary, urls, pdf_urls)` so the WebInsightGraph can both reason and show thumbnails. |

---

### 🗃️ local_db.py

| Section | Contents |
|---------|----------|
| **Imports / Globals** | `pandas`, `sqlalchemy`, `json`, `Path`; LangChain `Chroma`, `Document`, `SQLDatabase`; own `get_embeddings()`. Dummy CSV path `docs/dummy_lit.csv`. |
| **_make_fake_dataset()** | Writes a tiny JSON-driven “paper” set to CSV so the repo is self-contained. |
| **PaperMetaDB** | CSV → **SQLite** ingest (`papers` table) and wraps it in a LangChain `SQLDatabase`. |
| **PaperRAG** | Builds a **Chroma** index over `title + abstract`; exposes `get_docs()` for semantic search. |
| **PaperStore** | *Unifying façade* that keeps CSV + SQLite + Chroma in sync.<br>• `bootstrap()` → initial ingest.<br>• `add_paper()` appends one record everywhere.<br>• `make_local_search_tool()` returns an LC tool that performs Chroma similarity search. |
| **__main__ demo** | Shows: (1) SQL count per gene, (2) semantic search, (3) live add-paper and re-query. |



## ❓ FAQ <a id="faq"></a>
| Question | Answer |
| --- | --- |
| **Do I need an OpenAI key?** | Yes – set `OPENAI_API_KEY` in `.env`.                                                                 |
| **How do I add more tools?** | Create a LangChain `StructuredTool` and register it in `app.py`; the agent picks it up automatically. |
| **Poppler/PDF errors on Windows?**  | Install Poppler and add it to `PATH`; see the comments in `app.py` around `pdf2image`.                |

**📬  [Reach out to me](mailto:nekouvag@usc.edu)**
