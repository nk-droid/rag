import logging
from collections import deque
from datetime import datetime
from threading import RLock
from typing import Callable

from rich.markup import escape

_MAX_LOGS = 3
_LOCK = RLock()
_LOGS: deque[str] = deque(maxlen=_MAX_LOGS)
_REFRESH_CALLBACK: Callable[[], None] | None = None

def clear_logs() -> None:
    with _LOCK:
        _LOGS.clear()

def get_recent_logs() -> list[str]:
    with _LOCK:
        return list(_LOGS)

def set_refresh_callback(callback: Callable[[], None] | None) -> None:
    global _REFRESH_CALLBACK
    with _LOCK:
        _REFRESH_CALLBACK = callback

def add_log(message: str) -> None:
    callback: Callable[[], None] | None
    with _LOCK:
        _LOGS.append(message)
        callback = _REFRESH_CALLBACK
    if callback is not None:
        callback()

class RecentLogsHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level_style = {
            logging.DEBUG: "dim",
            logging.INFO: "cyan",
            logging.WARNING: "yellow",
            logging.ERROR: "red",
            logging.CRITICAL: "bold red",
        }.get(record.levelno, "white")

        level = escape(record.levelname)
        name = escape(record.name)
        message = escape(record.getMessage())
        ts = datetime.now().strftime("%H:%M:%S")
        add_log(f"[dim][{ts}][/dim] [{level_style}]{level}[/{level_style}] {name}: {message}")
