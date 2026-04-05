# Tutor Module Improvements

---

## Backend (agent/tutor.py)

### RAG Integration
- Added `_fetch_rag_context()` to pull RAG document excerpts into problem generation

### Math Verification
- `_verify_math_mcq()` — second-pass MCQ answer verification
- `_verify_math_solve()` — second-pass solve/proof answer verification

### JSON Parsing
- `_regex_extract_json_fields()` — Strategy 5 regex fallback for JSON parsing (when LLM output isn't valid JSON)

### Problem Generation
- Updated `generate_math_problem()` with RAG context + verification for ALL styles
- Fixed `get_math_step_by_step()` — generates steps via LLM when stored steps are empty

### Expression Safety
- Fixed blocked keyword check: `cos(x)` was blocked because `os` substring matched
- Changed to word-boundary regex for blocked words, substring only for `__`

### Solve Prompt
- Enhanced with CRITICAL INSTRUCTIONS for arithmetic verification
- Expression normalization: `2x` → `2*x`, `sinx` → `sin(x)`, `cos x` → `cos(x)`

---

## Dual LLM Model Switching (Session 3)
- `brain/config.py`: Added `MATH_LLM_MODEL` env var for separate math model (qwen3.5-27B)
- `agent/tutor.py`: `_llm(temperature, is_math=False)` selects MATH_LLM_MODEL for math tasks
- 5 math-specific calls updated: generate_math_problem, _generate_math_lesson, check_math_answer, get_math_step_by_step, _symbolic_derivative

---

## Curriculum System (Session 3)
- `MATH_CURRICULUM` data structure: 6 subjects × 4-6 chapters × 4-6 topics
  - **Subjects**: Algebra, Trigonometry, Calculus, Linear Algebra, Probability & Stats, Discrete Math
  - Functions: `get_curriculum()`, `get_chapter_topics()`, `mark_chapter_complete()`
  - In-memory progress tracking via `_curriculum_progress`
- API endpoints: GET `/math/curriculum`, GET `/math/curriculum/{subject}/{chapter}`, POST `/math/curriculum/progress`

---

## Frontend

### Tutor Container
- Width 94%, maxWidth 1400px, maxHeight 92vh (was 85%/1000px/80vh)

### Curriculum Browser
- "📚 Browse Topics" button → expandable subject/chapter/topic tree
- Click topic chip → auto-fills tutorTopic, enables math mode, closes browser
- Fetches from `/math/curriculum` on first open

### Math Tools Panel — 7 Interactive Visualizations
1. **Unit Circle**: angle slider, cos/sin/tan readouts, projections
2. **Vectors**: editable a⃗,b⃗ with sum, dot product, angle, parallelogram
3. **Triangle**: editable vertices, auto-computed sides/angles/area
4. **Circle**: editable center+radius, area/circumference/equation
5. **Matrix Transform**: editable 2×2 matrix, visual transform, eigenvalues, det, presets
6. **Normal Distribution**: μ/σ sliders, PDF curve, 1σ shaded region, statistics
7. **Bezier Curve**: t slider, De Casteljau construction, editable control points, formula

### Other UI
- Graph button auto-fetches on first toggle (no more blank graph)
- Stop button, green pulse, solve/proof input, g(x), axis labels, integral symbol

---

## Log Streaming
- Converted `brain/fast_search.py` print() → logging
- Converted `agent/code_context_builders.py` [RAG] print() → logger.info()
- Added "fast_search", "rag_brain", "query_pipeline", "code_context" to SSE logger list

---

## Key Bugs Found
- `cos(x)` blocked: substring "os" matched blocked keywords → use word-boundary regex
- Steps empty on JSON parse failure: fallback dict had `steps: []` → generate via LLM
- Python launcher: use `py` not `python` on this machine

## Key Learnings
- Always use word-boundary matching for blocked keywords, not substring
- Second-pass verification catches LLM math errors before showing to student
- Dual LLM setup (small fast model + large math model) gives best of both worlds
- 7 interactive visualizations cover most math concepts students encounter
