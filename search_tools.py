from __future__ import annotations
import os, time, requests
from typing import List, Dict, Tuple
from pathlib import Path
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from local_db import PaperStore
from models import get_embeddings, get_chat_model
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from sqlalchemy import text
import arxiv

def _throttle(min_interval=1.0):
    """Sleep so we don't hammer PubMed without API key (1 req/s )."""
    time.sleep(min_interval)

class WebSearch:
    """
    Search Google / Semantic Scholar, return structured hits,
    and optionally save them into the local PaperStore.
    """

    # ==== inner schema for LangChain =====
    class QueryInput(BaseModel):
        query: str = Field(..., description="Biomedical search string")
        top_k: int = Field(3, description="How many hits to fetch")


    def __init__(self, store: PaperStore):
        load_dotenv()
        self.store = store
        self.S2_KEY= os.getenv("S2_KEY")
        self.SERP_KEY= os.getenv("SERP_API_KEY")


    def _google_serp_search(self, query: str, k: int = 1) -> Tuple[str, List[Dict]]:
        """Return (text_context , structured_hits) from SerpAPI Google search."""
        url = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": query, "num": k, "api_key": self.SERP_KEY}
        _throttle()
        res  = requests.get(url, params=params, timeout=10).json()
        results = res.get("organic_results", [])[:k]

        # 1) text context for the LLM
        context_blocks = [
                "\n".join([
                f"{self._pick_emoji(r.get('link',""))} Title: {r.get('title','')} ({r.get('year','')})\n",
                f"Authors: {r.get('author','')}\n",
                f"Snippet: {(r.get('snippet') or "")[:800]}\n",
                f"Source: {(r.get('source') or "")}\n",
                f"URL: {r.get('link',"")}\n"
            ])
            for r in results
        ]

        context_text = "\n---\n".join(context_blocks)

        # 2) structured hits for PaperStore
        structs = [{
            "pmid":"", "title": r.get("title","")[:512], "authors":"", "journal":"",
            "year":None, "gene":"", "variant":"", "disease":"", "study_type":"",
            "n_patients":None, "effect_size":None,
            "abstract": r.get("snippet","")[:1000],
            "url": r.get("link","")
        } for r in results]

        return context_text, structs


    def _s2_api_search(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Hit Semantic Scholar Graph API and return (text_context, structured_hits)."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(k,100),
            "fields": "title,year,url,abstract,authors",
        }
        headers = {"x-api-key": self.S2_KEY}
        _throttle()
        res = requests.get(url, params=params, headers=headers, timeout=10).json()
        data = res.get("data", [])[:k]

        # 1) text context
        context_blocks = [
            "\n".join([
                f"{self._pick_emoji(p.get('url',""))} Title: {p.get('title','')} ({p.get('year','')})\n",
                f"Authors: {', '.join([author.get('name','') for author in p.get('authors','')])}\n",
                f"Snippet: {(p.get('abstract') or "")[:800]}\n",
                f"URL: {p.get('url',"")}\n"
            ])
            for p in data
        ]
        context_text = "\n---\n".join(context_blocks)

        # 2) structured
        structs = [{
            "pmid":"", "title": p.get("title","")[:512],
            "authors": "; ".join(a["name"] for a in p.get("authors",[])[:20]),
            "journal":"", "year": p.get("year"),
            "gene":"", "variant":"", "disease":"", "study_type":"",
            "n_patients":None, "effect_size":None,
            "abstract": (p.get("abstract") or "")[:4000],
            "url": p.get("url","")
        } for p in data]

        return context_text, structs


    def _arxiv_search(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Return (context_text, structured_hits) from the arXiv API."""
        client  = arxiv.Client()
        search  = arxiv.Search(query=query, max_results=k, sort_by=arxiv.SortCriterion.Relevance)
        results = list(client.results(search))[:k]
        
        # 1) text context
        context_blocks = [
            "\n".join([
                f"{self._pick_emoji(p.entry_id)} Title: {p.title} ({p.published.year})\n", # type: ignore
                f"Authors: {"; ".join(a.name for a in p.authors)}\n",
                f"Snippet: {(p.summary or '')[:800]}\n",
                f"URL: {p.entry_id}\n", 
            ])
            for p in results
        ]
        context_text = "\n---\n".join(context_blocks)

        # 2) structured hits
        structs = [{
            "pmid": "",
            "title": p.title[:512],
            "authors": "; ".join(a.name for a in p.authors[:3]),
            "journal": "arXiv",
            "year": p.published.year,
            "gene": "", "variant": "", "disease": "", "study_type": "",
            "n_patients": None, "effect_size": None,
            "abstract": (p.summary or "")[:4000],
            "url": p.entry_id
        } for p in results]

        return context_text, structs


    def _is_duplicate(self, store: PaperStore, title: str) -> bool:
        if not title.strip():
            return True            # treat blank titles as duplicates/skip
        q = text("SELECT 1 FROM papers WHERE title=:title LIMIT 1")
        with store.meta.engine.begin() as conn: # type: ignore
            return conn.execute(q, {"title": title}).fetchone() is not None


    def _pick_emoji(self, url: str) -> str:
        if "arxiv" in url:
            return "📄"
        if "nature" in url or "science.org" in url:
            return "🔬"
        if "bioRxiv" in url or "medrxiv" in url:
            return "🧬"
        return "📚"


    def _add_structs(self, store: PaperStore, structs):
        for p in structs:
            title = p.get("title") or ""
            if self._is_duplicate(store, title):
                continue
            store.add_paper(p)
            

    def fetch_all(self, query: str, k_google:int=1, k_s2:int=3) -> str:
        store = PaperStore(Path("docs/dummy_lit.csv"), embedding_model=get_embeddings())
        store.bootstrap()

        ctx_g, st_g = self._google_serp_search(query, k_google)
        ctx_s, st_s = self._s2_api_search(query, k_s2)
        ctx_a, st_a = self._arxiv_search(query, 5)   

        self._add_structs(store, st_g + st_s + st_a)

        # return "\n=== GOOGLE ===\n" + ctx_g + "\n\n=== S2 ===\n" + ctx_s
        return (
            "\n=== GOOGLE ===\n"  + ctx_g +
            "\n\n=== S2 ===\n"    + ctx_s +
            "\n\n=== ARXIV ===\n" + ctx_a
        )


    def make_combined_search_tool(self, store: PaperStore):
        # Search and Cache
        def _run(query: str, k_google:int=1, k_s2:int=1, k_arxiv:int=5):
            ctx_g, st_g = self._google_serp_search(query, k_google)
            ctx_s, st_s = self._s2_api_search(query, k_s2)
            ctx_a, st_a = self._arxiv_search(query, k_arxiv)
            
            self._add_structs(store, st_g + st_s + st_a)


            # collect ALL urls from the  hits
            urls = [p.get("url", "") for p in st_g + st_s + st_a if p.get("url")]
            pdf_urls = []
            for u in urls:                 # e.g. https://arxiv.org/abs/2502.05489
                if "arxiv" in u:
                    pdf_urls.append(u.replace("/abs/", "/pdf/"))
                elif u.endswith(".pdf"):   # plain-pdf from elsewhere
                    pdf_urls.append(u)


            # summary ="\n=== GOOGLE SCHOLAR ===\n" + ctx_g + "\n\n=== S2 ===\n" + ctx_s
            summary = "\n=== GOOGLE SCHOLAR ===\n"  + ctx_g +  "\n\n=== S2 ===\n" + ctx_s + "\n\n=== ARXIV ===\n" + ctx_a
            return summary, urls, pdf_urls

        class In(BaseModel):
            query: str = Field(..., description="Search phrase")
            k_google: int = Field(1, description="Google Scholar hits")
            k_s2: int = Field(1, description="Semantic Scholar hits")
            k_arxiv: int = Field(5, description="arXiv hits")

        return StructuredTool.from_function(
            name="Google_and_Semantic_scholar_search",
            description="Google-plus-SemanticScholar search; returns summary & caches papers.",
            func=_run, args_schema=In, return_direct=True)


if __name__=="__main__":
        
    store = PaperStore(Path("docs/dummy_lit.csv"), embedding_model=get_embeddings())
    store.bootstrap()
    local_tool = store.make_local_search_tool()

    web = WebSearch(store)
    search_tool = web.make_combined_search_tool(store)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a biomedical research assistant."),
        ("user", "{input}"), MessagesPlaceholder("agent_scratchpad")
    ])

    llm    = get_chat_model("big")

    agent  = create_openai_functions_agent(llm, [local_tool, search_tool], prompt)
    runner = AgentExecutor(agent=agent, tools=[local_tool, search_tool], verbose=True)

    runner.invoke({"input": "Which BRCA1 frameshift variants are linked to hereditary breast cancer? search online"})







