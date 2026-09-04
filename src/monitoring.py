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
        
        trace = self.client.trace(name=name, metadata=metadata or {})
        try:
            yield trace
        except Exception as e:
            trace.span(name="error", input={"error": str(e)})
            raise
        finally:
            self.client.flush()
    
    def score(self, trace_id: str, name: str, value: float):
        """Record a score."""
        if self.client:
            self.client.score(trace_id=trace_id, name=name, value=value)


class NoOpTrace:
    """No-op trace when monitoring is disabled."""
    
    def set_attribute(self, key: str, value: Any):
        pass
    
    def span(self, **kwargs):
        pass
