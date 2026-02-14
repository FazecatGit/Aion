import json
from datetime import datetime
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
    RETRIEVAL_K, CHUNK_SIZE, CHUNK_OVERLAP
)
from .keyword_search import search_documents
from .pdf_utils import load_pdfs

def _write_index_metadata(doc_count: int) -> None:
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "data_dir": DATA_DIR,
        "doc_count": doc_count,
        "created_at": datetime.utcnow().isoformat() + "Z",
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


def _filter_docs(docs: list[Document], filters: dict | None) -> list[Document]:
    if not filters:
        return docs
    sources = set(filters.get("source", [])) if filters.get("source") else None
    pages = set(filters.get("page", [])) if filters.get("page") else None
    filtered = []
    for doc in docs:
        meta = doc.metadata or {}
        if sources is not None and meta.get("source") not in sources:
            continue
        if pages is not None and meta.get("page") not in pages:
            continue
        filtered.append(doc)
    return filtered


def ingest_docs():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DIR)
    _write_index_metadata(len(splits))
    print(f"Ingested {len(splits)} chunks.")

def hybrid_retrieval(question: str, k: int = RETRIEVAL_K, filters: dict | None = None):

    # Semantic search via FAISS
    _validate_index_metadata()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": max(k * 3, k)})
    semantic_docs = semantic_retriever.invoke(question)
    semantic_docs = _filter_docs(semantic_docs, filters)
    
    # Keyword search
    pdf_docs = load_pdfs(DATA_DIR)
    keyword_docs = search_documents(question, pdf_docs, n_results=max(k * 3, k))
    
    # Convert keyword results to Document objects
    keyword_docs_obj = [
        Document(page_content=doc['content'], metadata=doc['metadata'])
        for doc in keyword_docs
    ]
    
    # Combine and deduplicate (keep order of semantic results and then add unique keywords into results)
    seen = set()
    combined = []
    
    for doc in semantic_docs:
        key = _doc_key(doc)
        if key not in seen:
            combined.append(doc)
            seen.add(key)
    
    for doc in keyword_docs_obj:
        key = _doc_key(doc)
        if key not in seen:
            combined.append(doc)
            seen.add(key)
    
    return combined[:k]

def query_brain(question: str, verbose: bool = False):
    docs = hybrid_retrieval(question)

    if verbose:
        print("RETRIEVED:", len(docs))
        for d in docs:
            print(d.metadata)
            print(d.page_content[:300])
            print("----")

    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    # Format context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    result = STRICT_RAG_PROMPT.invoke({"context": context, "input": question})
    llm_output = llm.invoke(result.to_string())
    
    return llm_output

if __name__ == "__main__":
    print("1: Ingest | 2: Query")
    choice = input("Choose: ")
    if choice == "1":
        ingest_docs()
    else:
        print("Query mode. Type 'quit' to exit.\n")
        while True:
            q = input("Ask: ").strip()
            if q.lower() == "quit":
                print("Goodbye!")
                break
            if q:
                result = query_brain(q)
                print(f"\nAnswer: {result}\n")
