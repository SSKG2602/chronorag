from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

try:  # Optional OpenTelemetry import
    from opentelemetry import trace
except Exception:  # pragma: no cover
    trace = None


@contextmanager
def traced_span(name: str) -> Iterator[None]:
    if trace:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(name):
            yield
    else:
        yield
