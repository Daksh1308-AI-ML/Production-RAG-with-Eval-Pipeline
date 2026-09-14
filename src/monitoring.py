"""Langfuse monitoring and tracing."""

from typing import Optional, Dict, Any
from contextlib import contextmanager

from .config import config


class Tracer:
    """Langfuse tracer for monitoring."""

    def __init__(self):
        self.client = None
        if config.monitoring.enabled:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=config.monitoring.public_key,
                    secret_key=config.monitoring.secret_key,
                    host=config.monitoring.host
                )
            except Exception as e:
                print(f"Langfuse initialization failed: {e}")

    @contextmanager
    def trace(self, name: str, metadata: Optional[Dict] = None):
        """Context manager for tracing."""
        if not self.client:
            yield NoOpTrace()
            return

        span = self.client.start_observation(name=name, metadata=metadata or {})
        adapter = _SpanAdapter(span, dict(metadata or {}))
        try:
            yield adapter
        except Exception as e:
            span.update(level="ERROR", status_message=str(e))
            raise
        finally:
            span.end()
            self.client.flush()

    def score(self, trace_id: str, name: str, value: float):
        """Record a score."""
        if self.client:
            self.client.create_score(trace_id=trace_id, name=name, value=value)


class _SpanAdapter:
    """Bridges the v3-style set_attribute calls to a Langfuse v4 span."""

    def __init__(self, span, metadata: Dict):
        self._span = span
        self._metadata = metadata

    def set_attribute(self, key: str, value: Any):
        self._metadata[key] = value
        self._span.update(metadata=self._metadata)


class NoOpTrace:
    """No-op trace when monitoring is disabled."""

    def set_attribute(self, key: str, value: Any):
        pass

    def span(self, **kwargs):
        pass
