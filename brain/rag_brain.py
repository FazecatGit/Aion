from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompts import RAG_PROMPT

DATA_DIR = "./data"
FAISS_DIR = "./faiss_index"

def ingest_docs():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"Ingested {len(splits)} chunks.")

def query_brain(question: str):
    prompt = RAG_PROMPT
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    print("RETRIEVED:", len(docs))
    for d in docs:
        print(d.metadata)
        print(d.page_content[:300])
        print("----")

    llm = OllamaLLM(model="dolphin-llama3:8b", temperature=0)
    
    prompt = PromptTemplate.from_template(
        """You MUST ONLY answer using the provided context. Do NOT make up information.

Context:
{context}

Question: {input}

If the answer is not found in the context, respond EXACTLY with: "I don't have that information in the documents."
Answer:"""
    )
    
    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

if __name__ == "__main__":
    print("1: Ingest | 2: Query")
    choice = input("Choose: ")
    if choice == "1":
        ingest_docs()
    else:
        q = input("Ask: ")
        print(query_brain(q))
