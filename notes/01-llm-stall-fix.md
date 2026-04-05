# LLM Stall Fix — TimedLLM + Prompt Caps

## Problem
- LLM calls stalled for minutes during agent operations
- No visibility into what the LLM was doing (no timing, no prompt size info)
- `OllamaLLM` had `num_ctx=None, num_predict=None` — no limits
- Prompts included full file + RAG context + history + lint + memory — massive

## Root Cause
A single agent request triggers 10-20+ `llm.invoke()` calls with unbounded prompts/generation.

## Changes Made

### brain/config.py
- Added `TimedLLM` wrapper class: logs every invoke/ainvoke with call ID, model, prompt size, timing, slow call warnings (>60s)
- Added constants: `LLM_NUM_CTX=8192`, `LLM_NUM_PREDICT=4096`, `LLM_TIMEOUT_SECONDS=180`
- `make_llm()` now returns `TimedLLM(OllamaLLM(..., num_ctx=8192, num_predict=4096))`
- Thread-safe call counter, `_estimate_tokens()`, dual logging (logger + print for SSE)

### agent/code_agent.py
- `_MAX_CONTEXT_CHARS = 6000` — caps RAG context block
- `_MAX_SNIPPET_CHARS = 12000` — caps file snippet
- Prompt budget logging: snippet/ctx/instruction sizes

### api.py
- Added `"llm_calls"` to SSE log streaming logger list

## Validation
- TimedLLM tested: construction shows num_ctx/num_predict set correctly
- invoke/ainvoke produce timing logs: `[LLM #N] invoke START/DONE — Xs, output=N chars`
- End-to-end agent/edit test: 10 LLM calls, all 0.2-8.5s, total ~26s, no stalls
- Prompt sizes all <2000 tokens with caps active

## Key Learnings
- Always cap both context window and generation length for local LLMs
- Wrap LLM calls with timing instrumentation to detect stalls early
- Dual logging (structured logger + print) ensures visibility in both SSE streams and log files
