"""Keyword search functionality for PDFs using text processing and stemming."""
import string
import nltk
import math
import os
import pickle
from pathlib import Path
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords
from collections import defaultdict, Counter

# Cache directory for TF-IDF index
CACHE_PATH = Path(__file__).parent.parent / "cache"

# Auto-download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()

class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(set)  # token -> set of document IDs
        self.docmap = {} #document ID -> document

        self.index_path = CACHE_PATH / "index.pkl" 
        self.docmap_path = CACHE_PATH / "docmap.pkl"

        self.term_frequencies = defaultdict(Counter) # token -> frequency across all documents
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"

    def _add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for t in set(tokens):
            self.index[t].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        return sorted(list(self.index[term]))

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term should be a single token")
        return self.term_frequencies[doc_id][tokens[0]]
        
    def get_idf(self, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term should be a single token")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self, documents: list[dict]):
        for i, doc in enumerate(documents):
            doc_id = i 
            self.docmap[doc_id] = doc
            text = doc.get('content', '')
            self._add_document(doc_id, text)
    
    def save(self):
        """Save index to cache files."""
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
    
    def load(self):
        """Load index from cache files."""
        if all(p.exists() for p in [self.index_path, self.docmap_path, self.term_frequencies_path]):
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            return True
        return False
    
    def is_cached(self):
        """Check if index is cached."""
        return all(p.exists() for p in [self.index_path, self.docmap_path, self.term_frequencies_path])


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    stopwords = set(nltk_stopwords.words('english'))
    res = []

    def _filter(tok):
        tok = tok.strip()
        if tok and tok not in stopwords:
            return True
        return False

    for tok in text.split():
        if _filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)     
    return res

def has_matching_token(query_tokens: list[str], doc_tokens: list[str]) -> bool:
    for q in query_tokens:
        for d in doc_tokens:
            if q in d:
                return True
    return False

def search_documents(query: str, documents: list[dict], n_results: int = 5) -> list[dict]:
    """Search documents using TF-IDF ranking with caching."""
    if not documents:
        return []
    
    # Build or load index
    index = InvertedIndex()
    
    # Try to load from cache
    if index.load():
        # Verify cache is for these documents (check doc count)
        if len(index.docmap) != len(documents):
            index = InvertedIndex()
            index.build(documents)
            index.save()
    else:
        # Build and cache index
        index.build(documents)
        index.save()
    
    # Score each document by TF-IDF
    query_tokens = tokenize_text(query)
    scores = {}
    
    for doc_id in range(len(documents)):
        score = sum(index.get_tfidf(doc_id, token) for token in query_tokens)
        if score > 0:
            scores[doc_id] = score
    
    if not scores:
        return []
    
    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [index.docmap[doc_id] for doc_id, _ in ranked_ids[:n_results]]
