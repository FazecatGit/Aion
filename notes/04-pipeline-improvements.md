# Pipeline Improvements

---

## Session 3: Core Improvements

### 1. Static Analysis / Linting
- `brain/config.py`: Added `LANG_LINT_CMD` config (Go: `go vet`, Python: `ruff check`)
- `agent/code_context_builders.py`: Added `run_lint_checks()` function
  - Go: wraps standalone functions in temp package, runs `go vet`
  - Python: runs `ruff check` if available on PATH
  - Returns formatted lint notes injected into LLM prompt
- `agent/code_agent.py`: Imports and calls `run_lint_checks()` in `edit_code()`, adds to context_block
- **Keep existing syntax checks** — linting complements them, doesn't replace

### 2. Go Standalone Function Fix
- `agent/code_context_builders.py`: Added `_go_compile_check()` helper
  - Detects Go files without `package` declaration (LeetCode-style)
  - Wraps in temp dir with `package main` + dummy `main()` for `go build`
  - Adjusts line numbers by -2 offset, filters dummy main errors

### 3. Test Execution in Multi-Agent Pipeline
- `agent/multi_agent_graph.py`:
  - Added `test_cases` and `test_results` to `AgentState`
  - `execute_node`: when test_cases available, uses `fix_with_tests()` instead of `edit_code()`
  - `critique_node`: uses execution-based verdict when test_results exist
  - `discuss_node`: includes test execution data in planner/critic prompts
  - `run_multi_agent()`: accepts optional `test_cases` param
- Frontend passes test_cases to orchestrate endpoint when multi-agent mode + tests filled in

### 4. RAG-Assisted Error Resolution
- `agent/code_agent.py` `fix_with_tests()`:
  - After attempt >= 1 with failures, queries RAG with error-specific query
  - Uses `fast_topic_search()` with keyword reranking for speed
  - Injects relevant algorithm/pattern docs into augmented_instruction

### 5. Enhanced Multi-Agent Discussion
- `agent/multi_agent_graph.py` `discuss_node`:
  - Builds test_section from test_results when available
  - Injects actual execution data into both planner and critic prompts
  - Agents debate with real test data, not just abstract LLM opinions

---

## Session 4: Bug Fixes

### Bug 1: SEARCH/REPLACE rejection on small files
- Root cause: 15-line LeetCode files had every SEARCH/REPLACE rejected → always fell through to whole-function rewrite
- Fix: `is_oversized_block()` uses 0.95 ratio for files < 50 lines (was 0.6 always)

### Bug 2: Wrong language generation (Go→Python)
- Root cause: 7B model loses language constraint in long prompts, generates Python for Go files
- Fix: Language guards in whole-function rewrite, `_strategy_pivot()`, and `fix_with_tests()`
- `lang_directive` moved to START of rewrite prompts + `lang_reminder` added at END

### Bug 3: File pollution between sequential calls
- Root cause: fix_with_tests writes intermediate states to disk, multi-agent reads stale/corrupted state
- Fix: Standalone `fix_with_tests` endpoint saves+restores original file content

### Bug 4: Language mismatch in test runner
- Root cause: After LLM generates Python for .go file, test runner creates Go harness with Python inside
- Fix: `run_tests()` validates source language matches file extension

---

## Session 5: Math Tutor + Multi-file Bugs

### Bug 5: Multi-file editing ignores CPP when Go is main
- Root cause: Context file always got "keep consistent with main file changes" instead of actual task instruction
- Fix: Cross-language detection flag, sends original instruction + reference diff

### Bug 6: Function-level editing deletes all other functions
- Root cause: `extract_function()` returned `("", 0, len(lines))` when target not found → whole-file replaced
- Three-pronged fix: word-matching fallback, function count guard, post-edit safety check

---

## Session 6: Lang Filter + Hard Problems

### Lang filter universal doc bypass
- Root cause: Algorithm textbooks dropped by pre-filter because filenames contained language keywords
- Fix: UNIVERSAL_TOPICS set {"algorithms", "clean-code", "mathematics"} bypasses language filter

### Query expansion for algorithm problems
- Turns "Fancy sequence" → "lazy evaluation modular inverse prefix sum" for better RAG hits

### Cross-encoder reranking after RRF fusion
- Added rerank step between RRF fusion and chunk grading

### Problem → editorial pair ingestion
- `ingest_editorials(pairs)` function for bulk-ingesting problem/editorial text pairs

## Key Learnings
- Small files need different edit thresholds than large files
- Language constraints must appear at both START and END of prompts for small models
- Test execution data > LLM opinions in multi-agent discussions
- Universal topic bypass prevents useful algorithm docs from being filtered out
- Query expansion via LLM dramatically improves RAG recall for obscure problem titles
