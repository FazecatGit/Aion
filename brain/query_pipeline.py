from typing import Optional

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

from .keyword_search import tokenize_text
from .prompts import (
    build_query_expansion_prompt,
    build_query_rewrite_prompt,
    build_spell_correction_prompt,
)


def _safe_llm_invoke(llm: OllamaLLM, prompt: str) -> Optional[str]:
    try:
        output = llm.invoke(prompt)
    except Exception:
        return None

    if output is None:
        return None
    return str(output).strip()


def _build_query_llm(llm_model: str) -> OllamaLLM:
    return OllamaLLM(model=llm_model, temperature=0)


def _spell_correct_query(query: str, llm: OllamaLLM) -> str:
    corrected = _safe_llm_invoke(llm, build_spell_correction_prompt(query))
    return corrected if corrected else query


def _rewrite_query(query: str, llm: OllamaLLM) -> str:
    rewritten = _safe_llm_invoke(llm, build_query_rewrite_prompt(query))
    return rewritten if rewritten else query


def _expand_query(query: str, llm: OllamaLLM) -> str:
    expanded = _safe_llm_invoke(llm, build_query_expansion_prompt(query))
    return expanded if expanded else query


def enhance_query_for_retrieval(
    query: str,
    llm_model: str,
    enable_spell_correction: bool,
    enable_rewrite: bool,
    enable_expansion: bool,
    verbose: bool = False,
) -> str:
    if not any([enable_spell_correction, enable_rewrite, enable_expansion]):
        return query

    llm = _build_query_llm(llm_model)
    enhanced_query = query

    if enable_spell_correction:
        enhanced_query = _spell_correct_query(enhanced_query, llm)
    if enable_rewrite:
        enhanced_query = _rewrite_query(enhanced_query, llm)
    if enable_expansion:
        enhanced_query = _expand_query(enhanced_query, llm)

    if verbose and enhanced_query != query:
        print(f"Enhanced query: '{query}' -> '{enhanced_query}'")
    return enhanced_query


def _keyword_rerank_documents(docs: list[Document], query: str) -> list[Document]:
    query_tokens = set(tokenize_text(query))
    if not query_tokens:
        return docs

    scored_docs = []
    query_lower = query.lower()
    for doc in docs:
        content = doc.page_content or ""
        content_tokens = set(tokenize_text(content))
        overlap = len(query_tokens & content_tokens)
        score = overlap / max(1, len(query_tokens))
        if query_lower in content.lower():
            score += 0.25
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]


def _cross_encoder_rerank_documents(
    docs: list[Document],
    query: str,
    cross_encoder_model: str,
    verbose: bool = False,
) -> list[Document]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        if verbose:
            print("Cross-encoder unavailable; falling back to keyword rerank.")
        return _keyword_rerank_documents(docs, query)

    try:
        model = CrossEncoder(cross_encoder_model)
        pairs = [(query, (doc.page_content or "")[:2000]) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]
    except Exception:
        if verbose:
            print("Cross-encoder scoring failed; falling back to keyword rerank.")
        return _keyword_rerank_documents(docs, query)


def rerank_documents(
    docs: list[Document],
    query: str,
    method: str,
    cross_encoder_model: str,
    verbose: bool = False,
) -> list[Document]:
    method_normalized = (method or "none").strip().lower()
    if method_normalized == "none":
        return docs
    if method_normalized == "keyword":
        return _keyword_rerank_documents(docs, query)
    if method_normalized == "cross_encoder":
        return _cross_encoder_rerank_documents(docs, query, cross_encoder_model, verbose=verbose)

    if verbose:
        print(f"Unknown rerank method '{method}'. Skipping rerank.")
    return docs
