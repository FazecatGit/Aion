import asyncio
import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Literal, Optional

from brain.fast_search import initialize_bm25
from brain.ingest import ingest_docs
from brain.ingest import ingest_file
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


class FeedbackRequest(BaseModel):
    question: str
    prev_mode: Optional[Literal['auto', 'fast', 'deep', 'both']] = 'auto'

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
    print("[API] ingest called")
    global raw_docs
    docs, topic_synonyms = await ingest_docs()
    raw_docs = load_pdfs(DATA_DIR)
    return {"topics": list(topic_synonyms.keys())}


@app.post("/open_data_folder")
async def open_data_folder():
    print("[API] open_data_folder called")
    import os, subprocess, sys
    data_dir = DATA_DIR
    from pathlib import Path
    p = Path(str(data_dir))
    if not p.exists():
        return {"status": "error", "error": f"path not found: {p}"}
    try:
        if sys.platform.startswith('win'):
            try:
                os.startfile(str(p))
            except Exception as e_start:
                # fallback to explorer.exe
                try:
                    rc = subprocess.run(['explorer', str(p)], check=False)
                    return {"status": "opened_fallback", "path": str(p), "rc": rc.returncode, "note": "used explorer fallback", "start_error": str(e_start)}
                except Exception as e_ex:
                    return {"status": "error", "error": str(e_ex), "start_error": str(e_start), "platform": sys.platform}
        elif sys.platform.startswith('darwin'):
            subprocess.run(['open', str(p)], check=True)
        else:
            subprocess.run(['xdg-open', str(p)], check=True)
        return {"status": "opened", "path": str(p), "platform": sys.platform}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"status": "error", "error": str(e), "trace": tb, "platform": sys.platform, "path": str(p), "exists": p.exists(), "is_dir": p.is_dir(), "cwd": str(Path.cwd())}


@app.post('/upload_and_ingest')
async def upload_and_ingest(file: UploadFile = File(...)):
    # Accepts multipart file upload (form field 'file') and saves into DATA_DIR
    from pathlib import Path
    print("[API] upload_and_ingest called")
    try:
        upload = file
        filename = Path(upload.filename).name
        dest = Path(DATA_DIR) / filename
        
        if dest.exists():
            return {"status": "exists", "filename": filename}
        contents = await upload.read()
        with dest.open('wb') as f:
            f.write(contents)

        result = await ingest_file(str(dest))
        if result is None:
            return {"status": "exists", "filename": filename}
        docs, topics = result
        return {"status": "ingested", "filename": filename, "topics": list(topics.keys())}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post('/ingest_file')
async def ingest_file_endpoint(filename: str):
    print("[API] ingest_file called; filename=", filename)
    # filename relative to DATA_DIR
    try:
        result = await ingest_file(filename)
        if result is None:
            return {"status": "exists"}
        docs, topics = result
        return {"status": "ingested", "topics": list(topics.keys())}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    # Run deep pipeline for the given question (user indicated dislike)
    results = await query_brain_comprehensive(
        req.question,
        verbose=False,
        raw_docs=raw_docs,
        session_chat_history=session_chat_history,
        mode_override='deep'
    )
    return results

@app.post("/clear")
async def clear():
    session_chat_history.clear()
    return {"status": "cleared"}
