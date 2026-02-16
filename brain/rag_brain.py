import json
from datetime import datetime
from pathlib import Path
from functools import reduce
from operator import itemgetter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .prompts import RAG_PROMPT, STRICT_RAG_PROMPT
from .config import (
    DATA_DIR, FAISS_DIR, INDEX_META_PATH, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
    RETRIEVAL_K, CHUNK_SIZE, CHUNK_OVERLAP, FUSION_MODE, FUSION_ALPHA, FUSION_K_PARAM
)
from .keyword_search import search_documents
from .pdf_utils import load_pdfs

def compose(*fns):
    """Compose functions left-to-right: compose(f, g)(x) = g(f(x))"""
    return reduce(lambda f, g: lambda x: g(f(x)), fns, lambda x: x)


def pipe(value, *fns):
    """Pipe value through sequence of functions"""
    return reduce(lambda x, f: f(x), fns, value)

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


def _load_index_metadata() -> dict:
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_index_metadata() -> None:
    if not Path(INDEX_META_PATH).exists():
        raise ValueError("Index metadata not found. Run ingest_docs() to create the index.")
    meta = _load_index_metadata()
    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "data_dir": DATA_DIR,
    }
    mismatches = []
    for key, value in expected.items():
        if meta.get(key) != value:
            mismatches.append(f"{key}: expected '{value}', got '{meta.get(key)}'")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Index metadata mismatch. Re-ingest required. Details: {details}")


def _doc_key(doc: Document) -> tuple:
    meta = doc.metadata or {}
    source = meta.get("source")
    page = meta.get("page")
    if source is not None and page is not None:
        return (source, page)
    return (source, page, doc.page_content[:200])


def _match_filters(filters: dict | None) -> callable:
    if not filters:
        return lambda doc: True
    
    allowed_sources = set(filters.get("source", [])) if filters.get("source") else None
    allowed_pages = set(filters.get("page", [])) if filters.get("page") else None
    
    def predicate(doc: Document) -> bool:
        meta = doc.metadata or {}
        if allowed_sources is not None and meta.get("source") not in allowed_sources:
            return False
        if allowed_pages is not None and meta.get("page") not in allowed_pages:
            return False
        return True
    
    return predicate


def _filter_docs(docs: list[Document], filters: dict | None) -> list[Document]:
    return list(filter(_match_filters(filters), docs))


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    range_val = max_score - min_score
    return [(s - min_score) / range_val for s in scores]


def _assign_rank_scores(docs: list[Document], k_param: int = 60) -> dict:
    return {
        _doc_key(doc): 1.0 / (k_param + rank)
        for rank, doc in enumerate(docs, start=1)
    }


def _merge_score_dicts(*score_dicts) -> dict:
    result = {}
    for scores in score_dicts:
        for key, score in scores.items():
            result[key] = result.get(key, 0) + score
    return result


def _rrf_fusion(semantic_docs: list[Document], keyword_docs: list[Document], k_param: int = 60) -> list[Document]:
    semantic_scores = _assign_rank_scores(semantic_docs, k_param)
    keyword_scores = _assign_rank_scores(keyword_docs, k_param)
    rrf_scores = _merge_score_dicts(semantic_scores, keyword_scores)
    
    doc_map = {_doc_key(doc): doc for doc in semantic_docs + keyword_docs}
    sorted_keys = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)
    
    return [doc_map[key] for key, _ in sorted_keys]


def _weighted_fusion(semantic_docs: list[Document], keyword_docs: list[Document], alpha: float = 0.5) -> list[Document]:
    semantic_scores = _normalize_scores(list(range(len(semantic_docs), 0, -1)))
    keyword_scores = _normalize_scores(list(range(len(keyword_docs), 0, -1)))
    
    semantic_map = {
        _doc_key(doc): (doc, semantic_scores[i])
        for i, doc in enumerate(semantic_docs)
    }
    keyword_map = {
        _doc_key(doc): (doc, keyword_scores[i])
        for i, doc in enumerate(keyword_docs)
    }
    
    all_keys = set(semantic_map.keys()) | set(keyword_map.keys())
    blended = {
        key: (
            semantic_map[key][0] if key in semantic_map else keyword_map[key][0],
            alpha * semantic_map.get(key, (None, 0.0))[1] + (1 - alpha) * keyword_map.get(key, (None, 0.0))[1]
        )
        for key in all_keys
    }
    
    sorted_items = sorted(blended.items(), key=lambda x: x[1][1], reverse=True)
    return [doc for _, (doc, _) in sorted_items]


def _build_keyword_docs(keyword_results: list[dict]) -> list[Document]:
    return [
        Document(page_content=doc['content'], metadata=doc['metadata'])
        for doc in keyword_results
    ]


def _select_fusion(fusion_mode: str):
    return {
        "rrf": _rrf_fusion,
        "weighted": _weighted_fusion,
    }.get(fusion_mode)


def _fuse_results(semantic_docs: list[Document], keyword_docs: list[Document], mode: str, **kwargs) -> list[Document]:
    fusion_fn = _select_fusion(mode)
    if fusion_fn:

        if mode == "rrf":
            return fusion_fn(semantic_docs, keyword_docs, k_param=kwargs.get("k_param", 60))
        else:
            return fusion_fn(semantic_docs, keyword_docs, alpha=kwargs.get("alpha", 0.5))
    
    seen = set()
    result = []
    for doc in semantic_docs + keyword_docs:
        key = _doc_key(doc)
        if key not in seen:
            result.append(doc)
            seen.add(key)
    return result


def ingest_docs():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    splits = pipe(
        loader.load(),
        lambda docs: splitter.split_documents(docs)
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DIR)
    _write_index_metadata(len(splits))
    print(f"Ingested {len(splits)} chunks.")


def hybrid_retrieval(question: str, k: int = RETRIEVAL_K, filters: dict | None = None, 
                     fusion_mode: str = "rrf", alpha: float = 0.5, k_param: int = 60) -> list[Document]:
    
    _validate_index_metadata()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Pipeline: retrieve → filter → fuse → truncate
    semantic_docs = pipe(
        vectorstore.as_retriever(search_kwargs={"k": max(k * 3, k)}).invoke(question),
        lambda docs: _filter_docs(docs, filters)
    )
    
    keyword_docs = pipe(
        load_pdfs(DATA_DIR),
        lambda docs: search_documents(question, docs, n_results=max(k * 3, k)),
        _build_keyword_docs
    )
    
    fused_docs = _fuse_results(
        semantic_docs, 
        keyword_docs, 
        fusion_mode,
        alpha=alpha,
        k_param=k_param
    )
    
    return fused_docs[:k]

def _is_likely_hallucination(context: str, question: str, answer: str) -> bool:
    """Check if answer appears to be hallucinated (not grounded in context)"""
    # If context is empty, definitely hallucinated
    if not context.strip():
        return True
    
    # Check if answer contains explicit "I don't have" - that's honest
    if "don't have" in answer.lower() or "not found" in answer.lower():
        return False
    
    # Very basic check: if NO word from question appears in context, likely hallucinated
    question_words = set(w.lower() for w in question.split() if len(w) > 3)
    context_lower = context.lower()
    
    matching_words = sum(1 for word in question_words if word in context_lower)
    coverage = matching_words / len(question_words) if question_words else 0
    
    # If less than 20% of question keywords in context, likely hallucinated
    return coverage < 0.2


def query_brain(question: str, verbose: bool = False, fusion_mode: str = None, alpha: float = None, k_param: int = None):
    fusion_mode = fusion_mode or FUSION_MODE
    alpha = alpha if alpha is not None else FUSION_ALPHA
    k_param = k_param if k_param is not None else FUSION_K_PARAM
    
    docs = hybrid_retrieval(question, fusion_mode=fusion_mode, alpha=alpha, k_param=k_param)

    if verbose:
        print("RETRIEVED:", len(docs))
        for d in docs:
            print(d.metadata)
            print(d.page_content[:300])
            print("----")

    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    if verbose:
        print("\nCONTEXT PASSED TO LLM:")
        print(context[:500])
        print("\n---\n")
    
    result = STRICT_RAG_PROMPT.invoke({"context": context, "input": question})
    llm_output = llm.invoke(result.to_string())
    
    # Detect hallucinations
    if _is_likely_hallucination(context, question, llm_output):
        return "I don't have that information in the documents."
    
    return llm_output

if __name__ == "__main__":
    print("1: Ingest | 2: Query")
    choice = input("Choose: ")
    if choice == "1":
        ingest_docs()
    else:
        print("Query mode. Type 'quit' to exit.\nType 'verbose' to toggle debug info.\n")
        verbose = False
        while True:
            q = input("Ask: ").strip()
            if q.lower() == "quit":
                print("Goodbye!")
                break
            if q.lower() == "verbose":
                verbose = not verbose
                print(f"Verbose mode: {verbose}")
                continue
            if q:
                result = query_brain(q, verbose=verbose)
                print(f"\nAnswer: {result}\n")
