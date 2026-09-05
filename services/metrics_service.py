"""Lightweight process-local workflow metrics."""

from __future__ import annotations

from collections import Counter
from threading import Lock


class MetricsService:
    """Track counts and durations without external monitoring infrastructure."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: Counter[str] = Counter()
        self._durations: dict[str, list[float]] = {}

    def record_run(self, status: str, duration: float) -> None:
        """Record a completed workflow status and duration."""

        with self._lock:
            self._counts["total_runs"] += 1
            self._counts[f"{status.lower()}_runs"] += 1
            if status == "PUBLISHED":
                self._counts["published_posts"] += 1
            self._durations.setdefault("workflow", []).append(max(0.0, duration))

    def record_stage(self, stage: str, duration: float) -> None:
        """Record one stage duration."""

        with self._lock:
            self._durations.setdefault(stage, []).append(max(0.0, duration))

    def snapshot(self) -> dict[str, float | int]:
        """Return counts and average durations."""

        with self._lock:
            result: dict[str, float | int] = dict(self._counts)
            for stage, values in self._durations.items():
                result[f"average_{stage}_duration"] = round(sum(values) / len(values), 4)
            return result


metrics = MetricsService()