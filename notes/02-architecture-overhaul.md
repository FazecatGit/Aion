# Architecture Overhaul

---

## 1. LaTeX Sanitization
**File**: `agent/tutor.py`
- Added `_strip_latex()` function that converts LaTeX → plain text
- `_sanitize_problem()` now calls `_strip_latex()` on question, explanation, steps
- MCQ and solve prompts have explicit "FORMATTING RULES" banning LaTeX

---

## 2. Lesson Introductions
**File**: `agent/tutor.py`
- Rewrote `_generate_math_lesson` prompt to require real-world hooks, analogies, building from prior knowledge

---

## 3. CS Curriculum
**Files**: `agent/tutor.py`, `api.py`
- Added `CS_CURRICULUM` dict: 5 subjects (cs_fundamentals, dsa, oop_design, systems, senior_engineering)
- Junior → mid → senior progression
- Functions: `get_cs_curriculum()`, `get_cs_chapter_topics()`, `mark_cs_chapter_complete()`
- API endpoints: `/cs/curriculum`, `/cs/curriculum/{subject}/{chapter}`, `/cs/curriculum/progress`

---

## 4. TimedLLM Extraction
**Files**: `brain/config.py` → `brain/timed_llm.py`
- Moved `TimedLLM` class, `_next_call_id`, `_estimate_tokens`, counters to `brain/timed_llm.py`
- `config.py` now imports `TimedLLM` from `brain.timed_llm`

---

## 5. Pattern Classification Pre-pass
**File**: `agent/code_agent.py`
- Added `_classify_problem_pattern()` method on `CodeAgent`
- Returns PATTERN/INVARIANT/EDGE CASES/COMPLEXITY classification
- Injected into analysis prompts for both implement and fix modes

---

## 6. Two-stage Prompt Chain
**File**: `agent/code_agent.py`
- Analysis phase now uses `reasoning_llm` at temp=0.1 (creative reasoning)
- Edit phase stays at temp=0.0 (precise coding)

---

## 7. Clean-slate Rewrite
**File**: `agent/code_agent.py`
- stagnant_count >= 2 now does pattern-classified clean-slate rewrite
- Ignores broken code, generates from scratch using classification + trace data
- Falls back to old `_strategy_pivot` if clean-slate fails

---

## 8. RAG Cosine Similarity
**File**: `brain/ingest.py`
- All `Chroma.from_documents()` calls now use `collection_metadata={"hnsw:space": "cosine"}`
- Added algorithm sub-category metadata in `_enrich_metadata()` (binary-search, dp, two-pointer, etc.)
- **NOTE: Requires re-ingest (`force=True`) to take effect on existing data**

## Key Learnings
- Separate creative reasoning (temp=0.1) from precise coding (temp=0.0)
- Pattern classification as a pre-pass gives the LLM much better context for problem-solving
- Cosine similarity outperforms default L2 for code/text embeddings
