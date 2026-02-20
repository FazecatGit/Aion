"""Augmented generation features: answering, summarization, citations, and detailed explanations."""

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from .config import LLM_MODEL, LLM_TEMPERATURE


def _format_documents_for_prompt(documents: list[Document]) -> str:
    formatted = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:500]
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        formatted.append(f"[{i}] {content}\n(Source: {source})")
    return "\n\n".join(formatted)


def answer_question(query: str, llm_model: str = None, verbose: bool = False) -> str:

    from .rag_brain import hybrid_retrieval
    
    llm_model = llm_model or LLM_MODEL
    

    docs = hybrid_retrieval(query, verbose=verbose)
    if not docs:
        return "I don't have enough information in the provided documents."
    formatted_docs = _format_documents_for_prompt(docs)

    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""Answer the following question based on the provided documents.

Question: {query}

Documents:
{formatted_docs}

Instructions:
- Provide a direct, concise answer
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"

Answer:"""
    response = llm.invoke(prompt)
    return response


def summarize_documents(query: str, llm_model: str = None, verbose: bool = False) -> str:
    from .rag_brain import hybrid_retrieval
    
    llm_model = llm_model or LLM_MODEL
    
    docs = hybrid_retrieval(query, verbose=verbose)
    if not docs:
        return "I don't have enough information in the provided documents."
    formatted_docs = _format_documents_for_prompt(docs)
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""Provide a comprehensive summary of the following documents that addresses this query.

Query: {query}

Documents:
{formatted_docs}

Instructions:
- Create a cohesive summary combining information from all documents
- Use citations like [1], [2] to reference sources
- Highlight key points and insights
- Keep it information-dense but readable

Summary:"""
    response = llm.invoke(prompt)
    return response

from .rag_brain import hybrid_retrieval
    
    
def cite_documents(query: str, llm_model: str = None, verbose: bool = False) -> str:
    llm_model = llm_model or LLM_MODEL
    
    docs = hybrid_retrieval(query, verbose=verbose)
    if not docs:
        return "I don't have enough information in the provided documents."
    formatted_docs = _format_documents_for_prompt(docs)
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""Find and extract the most relevant citations for this query.

Query: {query}

Documents:
{formatted_docs}

Instructions:
- For each relevant passage, provide the quote and source reference
- Explain why each citation is relevant to the query
- Format as: "[Source #] Quote" followed by explanation
- Include at least 3-5 key citations

Citations:"""
    response = llm.invoke(prompt)
    return response


def detailed_answer(query: str, llm_model: str = None, verbose: bool = False) -> str:
    llm_model = llm_model or LLM_MODEL
    
    docs = hybrid_retrieval(query, verbose=verbose)
    if not docs:
        return "I don't have enough information in the provided documents."
    formatted_docs = _format_documents_for_prompt(docs)
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""Provide a detailed, comprehensive answer to this question.

Question: {query}

Documents:
{formatted_docs}

Instructions:
- Start with a brief introduction
- Provide a thorough, multi-paragraph answer
- Cover different angles and perspectives
- Reference sources where appropriate
- End with key takeaways or conclusions

Answer:"""
    response = llm.invoke(prompt)
    return response


def query_brain_comprehensive(query: str, llm_model: str = None, verbose: bool = False) -> dict:
    return {
        'answer': answer_question(query, llm_model=llm_model, verbose=verbose),
        'summary': summarize_documents(query, llm_model=llm_model, verbose=verbose),
        'citations': cite_documents(query, llm_model=llm_model, verbose=verbose),
        'detailed': detailed_answer(query, llm_model=llm_model, verbose=verbose)
    }
