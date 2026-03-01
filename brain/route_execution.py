from brain.fast_search import fast_topic_search

_confidence_history = []
_last_feedback = None 


def _update_confidence(value: float):
    _confidence_history.append(value)
    if len(_confidence_history) > 100:
        _confidence_history.pop(0)


def _dynamic_threshold():
    if not _confidence_history:
        return None
    return sum(_confidence_history) / len(_confidence_history)


def record_feedback(feedback: str):
    global _last_feedback
    _last_feedback = feedback.lower() if feedback else None
    if _last_feedback not in ['like', 'dislike']:
        _last_feedback = None


def get_feedback():
    """Get last recorded feedback."""
    return _last_feedback


def probe_confidence(query: str) -> float:
    results = fast_topic_search(query)
    print(f"[DEBUG] BM25 results count: {len(results)}")
    if results:
        print(f"[DEBUG] Top score: {results[0].metadata.get('bm25_score', 'N/A')}")
    
    # compute confidence based on the BM25 scores
    scores = [r.metadata.get("bm25_score", 0) for r in results[:5]]
    print(f"[DEBUG] All scores: {scores}")
    
    top = scores[0] if scores else 0
    avg = sum(scores) / len(scores) if scores else 0
    confidence = top / (avg + 1e-6)
    print(f"[DEBUG] Confidence: {confidence:.3f}")
    
    _update_confidence(confidence)
    return confidence

def route_execution_mode(query: str) -> str:
    confidence = probe_confidence(query)
    threshold = _dynamic_threshold()

    if _last_feedback == 'dislike':
        print("[ROUTER] User disliked last result → escalating to deep_semantic")
        return "deep_semantic"

    if threshold is None:
        return "deep"

    if confidence >= threshold:
        return "fast"

    return "deep"