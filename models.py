"""
Utility registry for all OpenAI models used in the project.

Exports two helper functions:
    get_embeddings()  -> OpenAIEmbeddings 
    get_chat_model(cfg) -> ChatOpenAI    

The module hides the boilerplate of reading API keys and keeps a single
instance of each object to avoid re-initialising sockets on every call.
"""

# ------------------------- imports ------------------------------------------
from __future__ import annotations

import os
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv


# ------------------------- internal helpers ---------------------------------
def _api_key() -> str:
    load_dotenv()
    """Return the OpenAI key or raise a clear error."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found."
        )
    return key


# ------------------------- public factories ---------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Return a *singleton* OpenAIEmbeddings object.

    Caches the instance with @lru_cache so subsequent imports across the code
    base reuse the same underlying HTTPS connection pool.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=_api_key(), # type: ignore
    )


@lru_cache(maxsize=2)
def get_chat_model(size: str = "big") -> ChatOpenAI:
    """
    Return a ChatOpenAI model.
    • The models are cached separately (maxsize=2) so both variants stay alive.
    • Streaming=True lets Gradio display tokens as they arrive.
    """
    model_name = "gpt-4o" if size == "big" else "gpt-4o-mini"
    return ChatOpenAI(
        model=model_name,
        temperature=0.0,
        max_retries=2,
        streaming=True,
        api_key=_api_key(), # type: ignore
    )


# ------------------------- quick smoke test ---------------------------------
if __name__ == "__main__":
    
    chat = get_chat_model("small")
    print(chat.invoke("Say hi in one short sentence."))
