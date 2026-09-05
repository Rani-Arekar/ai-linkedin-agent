"""Deterministic tests for bounded transient retry behavior."""

import pytest

from services.retry_service import retry_call


def test_transient_failure_retries_with_exponential_delays() -> None:
    calls = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary")
        return "ok"

    assert retry_call(operation, sleep=delays.append) == "ok"
    assert calls == 3
    assert delays == [0.1, 0.2]


def test_retry_limit_is_respected() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("temporary")

    with pytest.raises(TimeoutError):
        retry_call(operation, attempts=2, sleep=lambda delay: None)
    assert calls == 2


def test_non_retryable_error_is_not_retried() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("invalid request")

    with pytest.raises(ValueError):
        retry_call(operation, sleep=lambda delay: None)
    assert calls == 1