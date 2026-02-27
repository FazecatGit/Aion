from rank_bm25 import BM25Okapi
import json, pickle
from typing import Any, List, Dict
from collections import Counter
import re
import os
import shutil

from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from .utils import pipe
from .config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_META_PATH, LLM_MODEL
from .pdf_utils import load_pdfs


TOPIC_MAP_PATH = "cache/topic_map.json"

def _write_index_metadata(doc_count: int) -> None:
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "data_dir": DATA_DIR,
        "doc_count": doc_count,
        "created_at": datetime.now().isoformat() + "Z",
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

def extract_keywords_from_corpus(docs: List[Any]) -> Dict[str, int]:
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'of', 'in', 'to', 'for', 'with', 'by', 'from', 'as', 'at', 'this', 'that',
        'it', 'its', 'if', 'can', 'we', 'you', 'your', 'them', 'their', 'more', 'than',
        'also', 'anyway', 'anything', 'anything else', 'anyone', 'anything at all',
        'anything in the world', 'anything like that', 'anything more', 'anything other than',
        'anything similar', 'anything to', 'anything under any circumstances', 'anything whatever',
        'anything within reason', 'anything you can think of', 'anything you could imagine',
        'anything whatsoever', 'anybody', 'anybody else', 'anybody in the world', 'anybody like that',
        'anybody more', 'anybody other than', 'anybody similar', 'anybody to', 'anybody under any circumstances',
        'anybody whatever', 'anybody within reason', 'anybody you can think of', 'anybody you could imagine',
        'anybody whatsoever', 
    }

    tokenized_docs = [doc.page_content.lower().split() for doc in docs]
    all_tokens = [token for doc in tokenized_docs 
                  for token in doc if token not in stop_words and len(token) > 2]
    
    word_freq = Counter(all_tokens).most_common(250)
    
    scores = [freq for _, freq in word_freq]
    
    return dict(zip([w for w, f in word_freq], scores))

async def ingest_docs(force: bool = False):
    """Ingest PDFs from DATA_DIR.

    If force is True, rebuilds from scratch (clears Chroma). Otherwise it will
    load any existing cached splits and only process new PDF files, appending
    their chunks to the existing index and adding new vectors to Chroma.
    """
    splits_path = Path("cache/splits.pkl")
    existing_splits = []
    if not force and splits_path.exists():
        try:
            with splits_path.open('rb') as f:
                existing_splits = pickle.load(f)
            print(f"Loaded {len(existing_splits)} existing chunks from cache")
        except Exception as e:
            print(f"Failed to load existing splits.pkl: {e}")
            existing_splits = []

    # load all pdf documents (langchain loader returns a list of Documents)
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    all_docs = loader.load()

    # determine which source files are already present
    existing_sources = set()
    for doc in existing_splits:
        md = getattr(doc, 'metadata', {})
        src = md.get('source') or md.get('file_path')
        if src:
            existing_sources.add(Path(src).resolve())

    # identify new docs (by source path)
    new_docs = []
    new_paths = []
    skipped_paths = []
    for doc in all_docs:
        md = getattr(doc, 'metadata', {})
        src = md.get('source') or md.get('file_path')
        resolved = Path(src).resolve() if src else None
        if resolved and resolved in existing_sources:
            skipped_paths.append(str(resolved))
            continue
        new_docs.append(doc)
        if resolved:
            new_paths.append(str(resolved))

    if new_paths:
        print("New files to ingest:")
        for p in new_paths:
            print(" -", p)
    if skipped_paths:
        print("Skipped (already ingested):")
        for p in skipped_paths:
            print(" -", p)

    if not new_docs and existing_splits and not force:
        print("No new documents to ingest; using existing index.")
        # still return loaded PDFs and topic map
        with open(TOPIC_MAP_PATH, 'r') as f:
            topic_map = json.load(f) if os.path.exists(TOPIC_MAP_PATH) else {}
        documents = load_pdfs(DATA_DIR)
        return documents, topic_map

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_splits = []
    if new_docs:
        new_splits = splitter.split_documents(new_docs)
    combined_splits = existing_splits + new_splits

    print(f"Loaded {len(new_splits)} new chunks (total {len(combined_splits)})")

    print("Extracting top keywords from corpus...")
    top_keywords = extract_keywords_from_corpus(combined_splits)
    print(f"Found {len(top_keywords)} key topics...")

    # rebuild BM25 from combined splits
    tokenized_docs = [doc.page_content.lower().split() for doc in combined_splits]
    bm25_idx = BM25Okapi(tokenized_docs)
    pickle.dump(bm25_idx, Path("cache/global_bm25.pkl").open('wb'))
    print(f"Built BM25 index: {len(combined_splits)} chunks")

    with open(TOPIC_MAP_PATH, 'w') as f:
        json.dump(top_keywords, f, indent=2)

    print("Updating Chroma vector DB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # prepare only the new splits for insertion to avoid re-embedding everything
    cleaned_new = []
    for doc in new_splits:
        clean_doc = doc.copy()
        clean_doc.metadata = {k: v for k, v in doc.metadata.items() if k not in ['source', 'file_path']}
        cleaned_new.append(clean_doc)

    if os.path.exists(CHROMA_DIR) and not force:
        if cleaned_new:
            # Add only new documents to existing Chroma store
            Chroma.from_documents(cleaned_new, embedding=embeddings, persist_directory=CHROMA_DIR)
            print(f"Appended {len(cleaned_new)} documents to Chroma at {CHROMA_DIR}")
        else:
            print("No new documents to add to Chroma.")
    else:
        # fresh build (either no CHROMA_DIR or force requested)
        if os.path.exists(CHROMA_DIR):
            print(f"Clearing old Chroma database at {CHROMA_DIR}...")
            shutil.rmtree(CHROMA_DIR)
        cleaned_all = []
        for doc in combined_splits:
            clean_doc = doc.copy()
            clean_doc.metadata = {k: v for k, v in doc.metadata.items() if k not in ['source', 'file_path']}
            cleaned_all.append(clean_doc)
        Chroma.from_documents(cleaned_all, embedding=embeddings, persist_directory=CHROMA_DIR)
        print(f"Built Chroma DB with {len(cleaned_all)} documents at {CHROMA_DIR}")

    # cache updated splits for BM25
    with Path("cache/splits.pkl").open('wb') as f:
        pickle.dump(combined_splits, f)
    print("✓ Cached splits.pkl for BM25")

    _write_index_metadata(len(combined_splits))

    documents = load_pdfs(DATA_DIR)

    print(f"Ingest complete: {len(combined_splits)} chunks → {len(documents)} PDFs, {len(top_keywords)} keywords, BM25 ready")
    return documents, top_keywords


async def ingest_file(file_path: str):
    """Ingest a single PDF file (path relative to DATA_DIR or absolute).

    Returns (documents, topic_map) similar to ingest_docs, or raises on error.
    """
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(DATA_DIR) / p

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # load existing splits if available
    splits_path = Path("cache/splits.pkl")
    existing_splits = []
    if splits_path.exists():
        try:
            with splits_path.open('rb') as f:
                existing_splits = pickle.load(f)
        except Exception:
            existing_splits = []

    # check if file already ingested by comparing source
    existing_sources = set()
    for doc in existing_splits:
        md = getattr(doc, 'metadata', {})
        src = md.get('source') or md.get('file_path')
        if src:
            try:
                existing_sources.add(Path(src).resolve())
            except Exception:
                pass

    if p.resolve() in existing_sources:
        return None  # indicate already exists

    # split the single document
    try:
        loader = PyPDFLoader(str(p))
        pages = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_splits = splitter.split_documents(pages)

    combined_splits = existing_splits + new_splits

    # update BM25
    tokenized_docs = [doc.page_content.lower().split() for doc in combined_splits]
    bm25_idx = BM25Okapi(tokenized_docs)
    pickle.dump(bm25_idx, Path("cache/global_bm25.pkl").open('wb'))

    # update Chroma with new splits only
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    cleaned_new = []
    for doc in new_splits:
        clean_doc = doc.copy()
        clean_doc.metadata = {k: v for k, v in doc.metadata.items() if k not in ['source', 'file_path']}
        cleaned_new.append(clean_doc)

    if os.path.exists(CHROMA_DIR):
        if cleaned_new:
            Chroma.from_documents(cleaned_new, embedding=embeddings, persist_directory=CHROMA_DIR)
    else:
        cleaned_all = []
        for doc in combined_splits:
            clean_doc = doc.copy()
            clean_doc.metadata = {k: v for k, v in doc.metadata.items() if k not in ['source', 'file_path']}
            cleaned_all.append(clean_doc)
        Chroma.from_documents(cleaned_all, embedding=embeddings, persist_directory=CHROMA_DIR)

    # cache updated splits
    with Path("cache/splits.pkl").open('wb') as f:
        pickle.dump(combined_splits, f)

    _write_index_metadata(len(combined_splits))

    documents = load_pdfs(DATA_DIR)

    top_keywords = extract_keywords_from_corpus(combined_splits)
    with open(TOPIC_MAP_PATH, 'w') as f:
        json.dump(top_keywords, f, indent=2)

    return documents, top_keywords