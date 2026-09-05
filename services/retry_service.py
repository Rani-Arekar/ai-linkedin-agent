"""Small bounded retry helper for transient external-service operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    initial_delay: float = 0.1,
    retry_exceptions: tuple[type[BaseException], ...] = (TimeoutError, OSError),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry only declared transient errors with bounded exponential backoff."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except retry_exceptions as error:
            last_error = error
            if attempt == attempts - 1:
                raise
            sleep(max(0.0, initial_delay) * (2**attempt))
    raise RuntimeError("Retry operation did not execute") from last_error