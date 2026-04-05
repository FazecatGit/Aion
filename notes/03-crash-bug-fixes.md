# Crash & Bug Fixes

---

## Session 6 — Multi-Agent Pipeline Fixes (6 issues)

1. **Strategy rejection**: critic_confidence in AgentState; escalate to strategist when ≤0.2
2. **ChromaDB overflow**: Truncate queries to 2000 chars before embedding
3. **Self-grader hardened**: Requires test-case trace before CORRECT/ISSUE verdict
4. **SEARCH/REPLACE skip**: format_failed flag → direct whole-function rewrite
5. **Go package validation**: Recover package declaration after rewrite
6. **Strategist RAG bias**: Removed planner's approach from RAG query

---

## Agent Debugging Fixes

### Problem
Code agent couldn't fix out-of-bounds array access bugs that manifest as "stack overflow" or "segfault" on LeetCode. Three root causes:
1. **Debug trace skipped on crashes**: `fix_with_tests` only ran debug trace when `actual is not None`. Crashes return `actual=None`.
2. **No ASan for C/C++**: Test harness compiled with `-O2` — crashes gave unhelpful errors.
3. **No crash-specific analysis**: Prompts had no guidance to check array bounds on crash bugs.

### Files Modified
- `agent/test_runner.py`: Added ASan rerun on C/C++ crashes, crash detection in `build_test_failure_note`, bounds-checking steps
- `agent/code_agent.py`: Crash error as debug_trace context, crash analysis checklist in analysis prompt

### Key Pattern
- `_crash_keywords` set in code_agent.py detects crash-type instructions

---

## Step Verification System

Agent failed on retries because it never EXECUTED its reasoning — just guessed.

### Architecture
- `run_step_verification()` in test_runner.py: generates assertion harness, runs it, parses results
- `_generate_assertion_harness()`: LLM generates program that checks intermediate values step-by-step
- `_parse_step_verification()`: parses STEP N: PASS/FAIL output into structured dict
- `format_step_verification_for_prompt()`: formats for injection into edit prompt

### Flow in fix_with_tests
1. Test fails → run_step_verification captures "Step 3 FAILED: got X, expected Y"
2. Step data → build_test_failure_note (self-reasoning + note output)
3. Note → augmented_instruction → edit_code fixes the specific failing step
4. Next iteration: step verification runs AGAIN to confirm the fix worked

### Key Details
- Step verification only runs on wrong-answer cases (actual != None), not crashes
- `all_crashes` flag in build_test_failure_note triggers bounds-focused reasoning
- ASan recompilation: `-O0 -g -fsanitize=address -fno-omit-frame-pointer`

## Key Learnings
- Always run debug traces even on crashes — `actual=None` is still valuable context
- ASan is essential for C/C++ debugging — catches buffer overflows that `-O2` hides
- Step-by-step verification beats "guess and retry" — forces the agent to actually execute reasoning
- Truncate ChromaDB queries to prevent embedding overflow
