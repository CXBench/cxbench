"""Retry interrupted evaluations without changing valid benchmark verdicts."""

from __future__ import annotations

import sys
import time
from threading import Event
from typing import Callable, TypeVar

from .openai_compat import ProviderTransportError


T = TypeVar("T")


def retry_until_completed(
    operation: Callable[[], T],
    *,
    label: str,
    retry_delay_seconds: float = 5.0,
    stop_event: Event | None = None,
) -> T:
    """Retry task errors; exhausted transport retries propagate to the caller."""
    attempt = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError(f"{label}: cancelled")
        attempt += 1
        try:
            return operation()
        except ProviderTransportError:
            raise
        except Exception as exc:
            print(
                f"{label}: attempt {attempt} failed ({type(exc).__name__}: {exc}); "
                f"retrying in {retry_delay_seconds:g}s",
                file=sys.stderr,
                flush=True,
            )
            if stop_event is None:
                time.sleep(retry_delay_seconds)
            elif stop_event.wait(retry_delay_seconds):
                raise InterruptedError(f"{label}: cancelled") from exc
