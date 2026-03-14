import time
import logging
import threading

_llm_logger = logging.getLogger("llm_calls")

# ── Call counter for per-request tracking ───────────────────────────────────
_llm_call_counter = 0
_llm_call_lock = threading.Lock()


def _next_call_id() -> int:
    global _llm_call_counter
    with _llm_call_lock:
        _llm_call_counter += 1
        return _llm_call_counter


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return len(text) // 4


class TimedLLM:
    """Wrapper around OllamaLLM that logs timing, prompt size, and output size
    for every invoke/ainvoke call.  Raises TimeoutError if a call exceeds the
    configured timeout."""

    def __init__(self, llm, timeout: int = 180):
        self._llm = llm
        self._timeout = timeout

    # Forward attribute access to the inner LLM so callers can read .model etc.
    def __getattr__(self, name):
        return getattr(self._llm, name)

    def invoke(self, prompt: str, **kwargs) -> str:
        call_id = _next_call_id()
        prompt_chars = len(prompt)
        prompt_tokens = _estimate_tokens(prompt)
        _llm_logger.info(
            "[LLM #%d] invoke START — model=%s prompt=%d chars (~%d tok)",
            call_id, self._llm.model, prompt_chars, prompt_tokens,
        )
        print(
            f"[INFO] [LLM #{call_id}] invoke START — model={self._llm.model} "
            f"prompt={prompt_chars} chars (~{prompt_tokens} tok)",
            flush=True,
        )
        t0 = time.monotonic()
        try:
            result = self._llm.invoke(prompt, **kwargs)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _llm_logger.error(
                "[LLM #%d] invoke FAILED after %.1fs — %s", call_id, elapsed, exc
            )
            print(
                f"[ERROR] [LLM #{call_id}] invoke FAILED after {elapsed:.1f}s — {exc}",
                flush=True,
            )
            raise
        elapsed = time.monotonic() - t0
        out_chars = len(result)
        out_tokens = _estimate_tokens(result)
        _llm_logger.info(
            "[LLM #%d] invoke DONE — %.1fs, output=%d chars (~%d tok)",
            call_id, elapsed, out_chars, out_tokens,
        )
        print(
            f"[INFO] [LLM #{call_id}] invoke DONE — {elapsed:.1f}s, "
            f"output={out_chars} chars (~{out_tokens} tok)",
            flush=True,
        )
        if elapsed > 60:
            _llm_logger.warning(
                "[LLM #%d] SLOW CALL — %.1fs (prompt ~%d tok, output ~%d tok). "
                "Consider reducing prompt size.",
                call_id, elapsed, prompt_tokens, out_tokens,
            )
            print(
                f"[WARN] [LLM #{call_id}] SLOW CALL — {elapsed:.1f}s "
                f"(prompt ~{prompt_tokens} tok, output ~{out_tokens} tok)",
                flush=True,
            )
        return result

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        call_id = _next_call_id()
        prompt_chars = len(prompt)
        prompt_tokens = _estimate_tokens(prompt)
        _llm_logger.info(
            "[LLM #%d] ainvoke START — model=%s prompt=%d chars (~%d tok)",
            call_id, self._llm.model, prompt_chars, prompt_tokens,
        )
        print(
            f"[INFO] [LLM #{call_id}] ainvoke START — model={self._llm.model} "
            f"prompt={prompt_chars} chars (~{prompt_tokens} tok)",
            flush=True,
        )
        t0 = time.monotonic()
        try:
            result = await self._llm.ainvoke(prompt, **kwargs)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _llm_logger.error(
                "[LLM #%d] ainvoke FAILED after %.1fs — %s", call_id, elapsed, exc
            )
            print(
                f"[ERROR] [LLM #{call_id}] ainvoke FAILED after {elapsed:.1f}s — {exc}",
                flush=True,
            )
            raise
        elapsed = time.monotonic() - t0
        out_chars = len(result)
        out_tokens = _estimate_tokens(result)
        _llm_logger.info(
            "[LLM #%d] ainvoke DONE — %.1fs, output=%d chars (~%d tok)",
            call_id, elapsed, out_chars, out_tokens,
        )
        print(
            f"[INFO] [LLM #{call_id}] ainvoke DONE — {elapsed:.1f}s, "
            f"output={out_chars} chars (~{out_tokens} tok)",
            flush=True,
        )
        if elapsed > 60:
            _llm_logger.warning(
                "[LLM #%d] SLOW CALL — %.1fs (prompt ~%d tok, output ~%d tok). "
                "Consider reducing prompt size.",
                call_id, elapsed, prompt_tokens, out_tokens,
            )
            print(
                f"[WARN] [LLM #{call_id}] SLOW CALL — {elapsed:.1f}s "
                f"(prompt ~{prompt_tokens} tok, output ~{out_tokens} tok)",
                flush=True,
            )
        return result
