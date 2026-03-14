"""
PrintLogger — thin wrapper around logging.Logger that also prints every log
message to stdout with flush=True so output appears in real-time even when
the frontend captures / buffers stderr or the logging stream.

Usage:
    from print_logger import get_logger
    logger = get_logger("code_agent")

All standard logger methods (.debug, .info, .warning, .error, .exception)
are forwarded to the real logger AND printed to stdout immediately.
"""

import logging
import sys


class PrintLogger:
    """Wraps a stdlib Logger — every call both logs AND prints (flush=True)."""

    def __init__(self, real_logger: logging.Logger):
        self._log = real_logger

    # ── forward attributes the rest of the codebase may touch ────────────
    @property
    def handlers(self):
        return self._log.handlers

    @property
    def level(self):
        return self._log.level

    def setLevel(self, level):
        self._log.setLevel(level)

    def addHandler(self, handler):
        self._log.addHandler(handler)

    def removeHandler(self, handler):
        self._log.removeHandler(handler)

    # ── core log-and-print methods ───────────────────────────────────────
    def _fmt(self, level: str, msg, args):
        try:
            formatted = msg % args if args else msg
        except (TypeError, ValueError):
            formatted = f"{msg} {args}"
        return f"[{level}] {formatted}"

    def debug(self, msg, *args, **kwargs):
        print(self._fmt("DEBUG", msg, args), flush=True)
        self._log.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        print(self._fmt("INFO", msg, args), flush=True)
        self._log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        print(self._fmt("WARN", msg, args), flush=True)
        self._log.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        print(self._fmt("ERROR", msg, args), flush=True)
        self._log.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        print(self._fmt("ERROR", msg, args), flush=True)
        self._log.exception(msg, *args, **kwargs)


def get_logger(name: str) -> PrintLogger:
    """Drop-in replacement for ``logging.getLogger(name)``."""
    return PrintLogger(logging.getLogger(name))
