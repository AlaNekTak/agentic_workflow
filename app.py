"""
UI backed by three LangGraph graph workflows:
  • WebInsightGraph   (search  ➜ scrape ➜ reflect)
  • SQLInsightGraph   (generate SQL ➜ execute ➜ summarize)
  • Local-RAG Graph   (local search ➜ reflect)
Requires:
  pip install langchain langgraph sqlalchemy pandas beautifulsoup4 fitz
  plus local modules: local_db.py, models.py, search_tools.py
"""

import requests, pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List
from pdf2image import convert_from_bytes
import json, io, base64, fitz , re, textwrap


from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from local_db   import PaperStore
from models     import get_embeddings, get_chat_model
from search_tools import WebSearch
# ────────────────────────────────────────────────────────────────────
### globals
store = PaperStore(Path("docs/dummy_lit.csv"), embedding_model=get_embeddings())
store.bootstrap()

web_helper   = WebSearch(store)
local_tool   = store.make_local_search_tool()
combined_web = web_helper.make_combined_search_tool(store)

llm_big  = get_chat_model("big")
llm_fast = get_chat_model("small")

# ─── HELPERS ─────────────────────────────────────────────────────────
# simple convenience wrappers
def local_ctx(q: str, k: int = 4) -> str:
    return "\n---\n".join(d.page_content for d in store.rag.get_docs(q, k))

def sql_df(sql: str) -> pd.DataFrame:
    with store.meta.engine.begin() as conn:              # type: ignore
        return pd.read_sql_query(sql, conn)

# ────────────────────────────────────────────────────────────────────
# ─── GRAPH 1 : WebInsightGraph (web only) ───────────────────────────
#  fetch_web ➜ scrape ➜ pdf_thumbs ➜ reflect
class WebState(TypedDict, total=False):
    question: str
    search_result: str
    scraped: str
    answer: str
    urls: List[str]
    pdf_urls: List[str]
    thumbs: List[tuple]

wg = StateGraph(WebState)  # type: ignore

# -- web search adapter ------------------------------------------------
@wg.add_node # type: ignore
def fetch_web(st: WebState) -> WebState:
    question = st["question"]           # type: ignore
    res, urls, pdf_urls  = combined_web.run({"query": question})
    return {"search_result": res, "urls": urls, "pdf_urls": pdf_urls}            # return update

@wg.add_node # type: ignore
def reflect(st: WebState) -> WebState:
    ctx = st.get("search_result", "") + "\n" + st.get("scraped", "")
    def safe_ctx(txt: str) -> str:
        return txt.replace("{", "{{").replace("}", "}}")
    safe_context = safe_ctx(ctx)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a biomedical assistant. Respond to the question using the retreived scholary work. "
        "You are provided with the title, snippet, and url and if available with the content of the papers. \n "
        "Always quote the exact paragraph that you used to infer the answer and provide that specific link."
        "Make sure to provide links to all references at the end of your answer. "
        "Like: References: link 1\n link2\n ..."),
        ("system", "CONTEXT:\n" + safe_context),
        ("user",   st["question"]) # type: ignore
    ])


    messages = prompt.format_messages()  
    answer = llm_big.invoke(messages).content # type: ignore
    return {"answer": answer}  # type: ignore

@wg.add_node  # type: ignore
def scrape(state: WebState) -> WebState:
    """
    Fetch **all** URLs in `state["urls"]`, concatenate the cleaned text
    (3000 chars per page max), and return it under `"scraped"`.
    """
    urls = state.get("urls", [])
    if not urls:
        return {}                         # nothing to fetch

    pages = []
    for url in urls:
        try:
            html  = requests.get(url, timeout=8).text
            text  = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)[:3000]
            pages.append(text)
        except Exception:
            continue                      # skip unreachable links

    if not pages:
        return {}                         # every fetch failed

    combined = "\n\nScraping URL:---\n\n".join(pages)  # separator between pages
    return {"scraped": combined}

@wg.add_node
def pdf_thumbs(state: WebState) -> WebState:
    thumbs = []

    for pdf_url in state.get("pdf_urls", []):
        try:
            raw = requests.get(pdf_url, timeout=15).content

            # ---------- primary renderer: pdf2image ----------
            try:
                img   = convert_from_bytes(raw, first_page=1, last_page=1, dpi=90)[0]
                thumbs.append((img, pdf_url))           # (image_obj, link_or_caption)
            except Exception:
                # ---------- fallback renderer: PyMuPDF ----------
                try:
                    doc  = fitz.open(stream=raw, filetype="pdf") # type: ignore
                    pix  = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(100/72, 100/72)) # type: ignore
                    b64  = base64.b64encode(pix.tobytes("png")).decode()
                except Exception:
                    # ---------- give up: use generic icon ----------
                    icon_path = Path(__file__).with_name("pdf_icon.png")
                    b64 = base64.b64encode(icon_path.read_bytes()).decode()

            thumbs.append(("data:image/png;base64," + b64, pdf_url))

        except Exception:
            # network failure → skip
            continue

    return {"thumbs": thumbs} if thumbs else {}

# entry/start → web, then scrape, then reflect
wg.add_edge(START, "fetch_web")
wg.add_edge("fetch_web", "scrape")
wg.add_edge("scrape", "pdf_thumbs")
wg.add_edge("pdf_thumbs",   "reflect")
wg.add_edge("reflect", END)
WebGraph = wg.compile()

# ────────────────────────────────────────────────────────────────────
# ─── GRAPH 2 : Local-RAG Graph (cached papers) ──────────────────────
# (local search ➜ reflect)

class RagState(TypedDict, total=False):
    question: str
    search_result: str
    answer: str
rg = StateGraph(RagState)  # type: ignore

# -- local search adapter -------------------------------------------
@rg.add_node  # type: ignore
def search_local(st: RagState) -> RagState: 
    res = local_tool.run({"query": st["question"]})  # type: ignore
    return {"search_result": res}

@rg.add_node  # type: ignore
def reflect_local(st: RagState) -> RagState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a biomedical assistant using ONLY the local paper cache. "
        "Please respond to user question based on retrieved documents. You may ask clarifying questions if anything is vague."
        "If the retrieved docs are not helping with the answer, refrain from answering based on your world-knowledge "
        "and advise the user to follow up with onine search. "
        "Make sure to report back the document meta data (e.g., Title, Year, URL) at the end of your answer for their reference."),
        ("system", "CONTEXT:\n" + st.get("search_result", "")),
        ("user",   st["question"]) # type: ignore
    ])
    messages = prompt.format_messages()
    answer = llm_big.invoke(messages).content  # type: ignore
    return {"answer": answer}  # type: ignore

rg.add_edge(START, "search_local")
rg.add_edge("search_local", "reflect_local")
rg.add_edge("reflect_local", END)
RAGraph = rg.compile()

# ────────────────────────────────────────────────────────────────────
# ─── GRAPH 3 : SQLInsightGraph ───────────────────────────────────────
#  write_sql ➜ run_sql ➜ | ➜ write_sql 
#                          | ➜ summarize 
class SQLState(TypedDict, total=False):
    question: str
    sql: str
    df_txt: str
    answer: str
    failed: bool
    last_error: str

sg = StateGraph(SQLState) # type: ignore

@sg.add_node # type: ignore
def write_sql(st: SQLState) -> SQLState:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Write ONE valid runnable SQLite query for table `papers`.\n"
         "Do not put ``` or extra characters, your SQL should be like 'SELECT * FROM papers' with no extra chars"
         "If you see `last_error`, fix the problem and try a different query.\n"
        f"last_error: {st.get('last_error', 'none')}"),
        ("user",   "{question}")
    ])
    messages = prompt.format_messages(question=st["question"])  # type: ignore
    raw = llm_big.invoke(messages).content# type: ignore
    sql = re.sub(r"```(?:sql)?|```", "", raw, flags=re.I).strip() # type: ignore
    sql = textwrap.dedent(sql)           # remove indent if any
    if sql.endswith(";"):
        sql = sql[:-1]
    return {"sql": sql, "question": st["question"]}  # type: ignore

@sg.add_node # type: ignore
def run_sql(st: SQLState) -> SQLState:
    try:
        df = pd.read_sql_query(st["sql"], store.meta.engine) # type: ignore
        return {"df_txt": df.to_markdown(index=False), "failed": False}
    except Exception as e:
        return {
            "failed": True,
            "last_error": str(e)[:400]     # keep it short
        }
    
@sg.add_node # type: ignore
def summarise(st: SQLState) -> SQLState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain these results briefly. And print the dataframe (if any) in Markdown"),
        ("system", st["df_txt"]), # type: ignore
        ("user",   st["question"]) # type: ignore
    ])
    messages = prompt.format_messages()
    answer = llm_big.invoke(messages).content # type: ignore
    return {"answer": answer}   # type: ignore

sg.add_edge(START, "write_sql")
sg.add_edge("write_sql", "run_sql")

# auto-retry on SQL error → feed last_error back into LLM
sg.add_conditional_edges(
    "run_sql",
    lambda st: "retry" if st.get("failed") else "done",
    {
        "retry": "write_sql",            # go back for a new SQL
        "done":  "summarise"             # continue the happy path
    }
)
sg.add_edge("summarise", END)
SQLGraph = sg.compile()

# ─── GRAPH WRAPPER TOOLS ────────────────────────────────────────────

def _run_graph(graph, payload):
    answer, thumbs, traces = "", [], []

    for chunk in graph.stream(payload, stream_mode="updates"):
        node_name, update = next(iter(chunk.items()))

        if "answer" in update:        # grab the answer when it appears
            answer = update["answer"]
        if "thumbs" in update:
            thumbs.extend(update["thumbs"])
            continue

        traces.append(f"[{node_name}]\n```json\n{json.dumps(update, indent=2)}\n```\n")
    return {"answer": answer, "thumbs": thumbs, "traces": traces}


# ─── SMART RESEARCH AGENT ────────────────────────────────────────────────────
def build_agent():
    class RunSQLIn(BaseModel):
        question: str = Field(
            ...,
            description="User’s **plain-English** question.  NOT raw SQL."
        )
    run_web_graph_tool = StructuredTool.from_function(
        name="run_web_graph",
        description="Do retrieval/scrape/reflect on a question. Returns {'answer','thumbs','traces'}.",
        func=lambda q: _run_graph(WebGraph, {"question": q}),
        return_direct=True, # no further processing 
    )

    run_sql_graph_tool = StructuredTool.from_function(
        name="run_sql_graph",
        description=(
            "Generate an SQLite query for table `papers`, run it, and return a short "
            "human-readable summary of the result.  ⚠️  Give me the English question, "
            "Next agent will write the SQL."
        ),
        args_schema=RunSQLIn,                         
        func=lambda question: _run_graph(
            SQLGraph, {"question": question}         
        ),
    )
    # Local-RAG wrapper ---------------------------------------------------
    run_local_rag_tool = StructuredTool.from_function(
        name="run_local_rag",
        description="Answer using ONLY the locally-cached literature (no internet).",
        func=lambda q: _run_graph(RAGraph, {"question": q})
    )
    
    SYSTEM = (
        "You are a helpful biomedical research assistant.\n"
        "• For everyday conversation: answer directly without calling tools when possible.\n"
        "• If the user asks a biomedical / scientific question that *mentions* or *implies* "
        "\"local\", \"cached\", or \"already downloaded\" literature → call **run_local_rag**.\n"
        "• Otherwise, for general biomedical / scientific queries that require external knowledge → "
        "call **run_web_graph**.\n"
        "• If the user asks for counts, aggregates, or other metrics stored in the local SQL "
        "database (table `papers`) → call **run_sql_graph**.\n"
        "Tools available:\n"
        "  - run_local_rag  : answer from local paper cache only\n"
        "  - run_web_graph  : web search + scraping + summarisation\n"
        "  - run_sql_graph  : generate & run SQL, then summarise numeric results\n"
    )

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("messages"),      # <<<
        ("assistant", "{agent_scratchpad}")
    ])
    tools  = [run_web_graph_tool, run_local_rag_tool, run_sql_graph_tool]
    agent  = create_openai_functions_agent(llm_big, tools, agent_prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
