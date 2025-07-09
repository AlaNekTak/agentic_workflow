"""
A micro-pipeline that bootstraps a Research-paper dataset into
(1) a SQLite metadata DB and (2) a Chroma vector store for RAG.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import json
from sqlalchemy import create_engine, Engine
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# ── local ────────────────────────────────────────────────────────
from models import get_embeddings

# ─────────────────────────────────────────────────────────────────
# Constants & paths
# -----------------------------------------------------------------
DOCS_DIR   = Path(__file__).parent / "docs"
DUMMY_CSV  = DOCS_DIR / "dummy_lit.csv"
DATA_JSON  = DOCS_DIR / "data.json"

def _make_fake_dataset(csv_path: Path) -> None:
    """Write demo biomedical-paper rows to *csv_path*."""
    try:
        with DATA_JSON.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Expected seed file {DATA_JSON}."
        ) from exc

    pd.DataFrame(data).to_csv(csv_path, index=False)

# ---------- 2. helper class:  SQLite warehouse -----------------
class PaperMetaDB:
    """Load a CSV into SQLite and expose a LangChain SQLDatabase."""

    def __init__(self, csv_path: Path, sqlite_path: Path = DOCS_DIR / "papers.db"):
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.engine: Engine | None = None
        self.db: SQLDatabase | None = None


    def ingest(self) -> SQLDatabase:
        """Read CSV → SQLite → wrap with LangChain SQLDatabase."""
        df = pd.read_csv(self.csv_path)
        self.engine = create_engine(f"sqlite:///{self.sqlite_path}")
        df.to_sql("papers", self.engine, if_exists="replace", index=False)
        self.db = SQLDatabase(engine=self.engine)
        return self.db


# ---------- 3. helper class:  Chroma retriever -----------------
class PaperRAG:
    """Build a Chroma vector store over `title+abstract`."""

    def __init__(self, csv_path: Path, embedding_model=None):
        self.csv_path = csv_path
        self.embedding_model = embedding_model or get_embeddings()
        self.vectorstore: Chroma | None = None

    def build(self) -> None:
        """Create the Chroma index from the CSV."""
        df = pd.read_csv(self.csv_path)

        docs: list[Document] = []
        for _, row in df.iterrows():
            # 1. what the retriever will see
            content = f"{row['title']} {row['abstract']}"

            # 2. what we keep as metadata
            meta = {
                "pmid": row["pmid"],
                "gene": row["gene"],
                "variant": row["variant"],
                "disease": row["disease"],
                "year": row["year"],
                "source": str(self.csv_path),
            }
            docs.append(Document(page_content=content, metadata=meta))

        # fresh build every time;
        self.vectorstore = Chroma.from_documents(
            docs,
            embedding=self.embedding_model,
            collection_name="lit_chroma",
            persist_directory=str(DOCS_DIR / "chroma_index"), # type: ignore
        )
        # self.vectorstore.persist()  

    def get_docs(self, query: str, k: int = 3):
        if self.vectorstore is None:
            raise RuntimeError("build() not called yet")
        return self.vectorstore.similarity_search(query, k=k)


# ---------- 4. unifying helper: PaperStore ------------------------------------
class PaperStore:
    """
    Convenience wrapper that keeps the metadata DB and the vector store in sync.

    Usage
    -----
    store = PaperStore(csv_path=DUMMY_CSV)
    store.bootstrap()                 # initial ingest + build
    store.add_paper({...})            # later, add new record retrieved online
    """

    # ---- static helpers - class-level metadata ---------------------------------
    @staticmethod
    def required_fields() -> list[str]:
        """Return the columns every paper dict must contain."""
        return [
            "pmid", "title", "authors", "journal", "year",
            "gene", "variant", "disease", "study_type",
            "n_patients", "effect_size", "abstract", "url"
        ]

    # ---- init / bootstrap ---------------------------------------------------
    def __init__(
        self,
        csv_path: Path,
        sqlite_path: Path | None = None, # Optional[Path] = None
        embedding_model=None,
    ):
        self.csv_path = csv_path
        self.embedding_model = embedding_model or get_embeddings()
        self.sqlite_path = sqlite_path or csv_path.with_suffix(".db")
        self.meta = PaperMetaDB(csv_path, self.sqlite_path)
        self.rag = PaperRAG(csv_path, self.embedding_model)
        self.bootstrap()

    def bootstrap(self) -> None:
        """Run ingest → build index (only once at start-up)."""
        self.meta.ingest()
        self.rag.build()

    # ---- dynamic update -----------------------------------------------------
    def add_paper(self, paper: dict) -> None:
        """
        Append a single paper (dict) to CSV, SQLite, and Chroma index.

        Parameters
        ----------
        paper : dict
            Must contain every key from `required_fields()`.
        """
        # 1. validate ---------------------------------------------------------
        missing = [f for f in self.required_fields() if f not in paper]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

        # 2. append to CSV ----------------------------------------------------
        df_new = pd.DataFrame([paper])
        df_new.to_csv(self.csv_path, mode="a", index=False, header=not self.csv_path.exists())

        # 3. upsert into SQLite ----------------------------------------------
        #    (Here: simple append; for real duplicates use ON CONFLICT clauses)
        with self.meta.engine.begin() as conn:  # type: ignore
            df_new.to_sql("papers", conn, if_exists="append", index=False)

        # 4. add to Chroma ----------------------------------------------------
        doc = Document(
            page_content=f"{paper['title']} {paper['abstract']}",
            metadata={
                "pmid": paper["pmid"],
                "gene": paper["gene"],
                "variant": paper["variant"],
                "disease": paper["disease"],
                "year": paper["year"],
                "source": str(self.csv_path),
            },
        )
        self.rag.vectorstore.add_documents([doc]) # type: ignore


    def make_local_search_tool(self):
        """
        Build a LangChain tool that queries the existing Chroma vector store
        and returns the top-k passages as plain text for the LLM context.
        """

        class In(BaseModel):
            query: str = Field(..., description="What to search for")
            k: int = Field(5, description="Number of passages")
        
        def _run(query: str, k: int = 5):
            docs = self.rag.get_docs(query)[:k]
            return "\n---\n".join(d.page_content for d in docs)
        
        return StructuredTool.from_function(
            func=_run, name="search_local_papers", args_schema=In, return_direct=True,
            description="Search only the locally-cached biomedical papers."
        )


# ------ 5. Tiny demo when executed directly -------------------------------------
if __name__ == "__main__":

    # ----------------------- thought process -----------------------
    # 1. Create a dummy DataFrame so the script is self-contained.
    # 2. Write it to CSV (stand-in for our harvested PubMed set).
    # 3. Build SQLite: simple one-table schema.
    # 4. Build Chroma: embed title+abstract; expose `get_docs`.
    # 5. Demo: run a semantic search and an SQL aggregation.
    # ---------------------------------------------------------------

    # ~~~ create dataset ~~~
    if not DUMMY_CSV.exists():
        _make_fake_dataset(DUMMY_CSV)
        print(f"Wrote {DUMMY_CSV}")

    # ~~~ SQLite path ~~~
    meta_db = PaperMetaDB(DUMMY_CSV).ingest()
    print("Tables:", meta_db.get_usable_table_names())
    # Example SQL: count papers per gene
    q = "SELECT gene, COUNT(*) AS n FROM papers GROUP BY gene;"
    print("SQL result →", meta_db.run(q))

    # ~~~ Chroma path ~~~
    rag = PaperRAG(DUMMY_CSV)
    rag.build()
    hits = rag.get_docs("frameshift BRCA1 variants")
    print("\nTop semantic hit(s):")
    for h in hits:
        print(">>", h.page_content[:120], "...")
        print("Metadata:", h.metadata)

    # ---- live update example -------------------------------------------------
    store = PaperStore(DUMMY_CSV, embedding_model=get_embeddings())

    new_record = {
        "pmid": "333333",
        "title": "Novel CFTR G551D corrector improves chloride conductance",
        "authors": "Zhang Q; Rivera L",
        "journal": "J Clin Invest",
        "year": 2023,
        "gene": "CFTR",
        "variant": "G551D",
        "disease": "cystic fibrosis",
        "study_type": "randomized trial",
        "n_patients": 56,
        "effect_size": 1.8,
        "abstract": "A small-molecule corrector targeting G551D increases channel open probability...",
        "url": "https://pubmed.ncbi.nlm.nih.gov/333333/"
    }

    store.add_paper(new_record)
    print("After update, semantic search for 'CFTR G551D':")
    for hit in store.rag.get_docs("CFTR G551D", k=2):
        print("-", hit.metadata["pmid"], hit.page_content[:80], "…")

