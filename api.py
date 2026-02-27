import asyncio
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Literal, Optional

from brain.fast_search import initialize_bm25
from brain.ingest import ingest_docs
from brain.augmented_generation_query import query_brain_comprehensive, session_chat_history
from brain.pdf_utils import load_pdfs
from brain.config import DATA_DIR

warnings.filterwarnings("ignore")

raw_docs = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global raw_docs
    raw_docs = load_pdfs(DATA_DIR)
    initialize_bm25(raw_docs)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str
    verbose: bool = False
    mode: Literal['auto', 'fast', 'deep', 'both'] = 'auto'

@app.post("/query")
async def query(req: QueryRequest):
    # pass along mode override; query_brain_comprehensive handles 'both'
    results = await query_brain_comprehensive(
        req.question,
        verbose=req.verbose,
        raw_docs=raw_docs,
        session_chat_history=session_chat_history,
        mode_override=req.mode
    )
    return results

@app.post("/ingest")
async def ingest():
    global raw_docs
    docs, topic_synonyms = await ingest_docs()
    raw_docs = load_pdfs(DATA_DIR)
    return {"topics": list(topic_synonyms.keys())}

@app.post("/clear")
async def clear():
    session_chat_history.clear()
    return {"status": "cleared"}
