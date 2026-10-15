"""Append-only Gemini debug log (timings, model, errors).

Log path is reported in the dialog; share this file if something hangs.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_LOG_PATH: Path | None = None


def set_log_path(path: Path) -> Path:
    global _LOG_PATH
    _LOG_PATH = Path(path)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _LOG_PATH


def log_path() -> Path:
    global _LOG_PATH
    if _LOG_PATH is None:
        # Default next to this package; dialog overrides to add-on folder
        _LOG_PATH = Path(__file__).resolve().parent / "gemini.log"
    return _LOG_PATH


def log(message: str, **fields: object) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
    extra = ""
    if fields:
        parts = [f"{k}={v!r}" for k, v in fields.items()]
        extra = " | " + " ".join(parts)
    line = f"{stamp} {message}{extra}\n"
    path = log_path()
    with _lock:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass


class timed:
    """Context manager that logs start/end with elapsed seconds."""

    def __init__(self, label: str, **fields: object) -> None:
        self.label = label
        self.fields = fields
        self.t0 = 0.0

    def __enter__(self) -> timed:
        self.t0 = time.monotonic()
        log(f"BEGIN {self.label}", **self.fields)
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        elapsed = time.monotonic() - self.t0
        if exc is not None:
            log(
                f"FAIL  {self.label}",
                elapsed_s=f"{elapsed:.2f}",
                error=str(exc),
                **self.fields,
            )
        else:
            log(f"END   {self.label}", elapsed_s=f"{elapsed:.2f}", **self.fields)
