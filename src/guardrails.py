"""Guardrails for prompt injection, PII, and low-value queries.

Input guardrail: block obvious prompt injections / PII / off-topic asks.
Output guardrail: refuse answers that leak PII or admit nothing found.

ponytail: keyword + regex based — a hosted moderation model is the upgrade
path if adversarial traffic ever appears; for a local demo these catch the
common cases in microseconds.
"""

import re
from typing import Optional

from .config import config

_REFUSAL = ("I can't answer that: it looks like a prompt injection, "
            "requests personal data, or is off-topic for SEC 10-K filings.")

_INJECTION_PATTERNS = (
    r"(?i)ignore (all |your |the )?(previous |above |prior )?instructions",
    r"(?i)disregard (the |your )?(previous |above )?(instructions|prompt|system)",
    r"(?i)you are now (a|acting as)",
    r"(?i)pretend (you are|to be) (dan|jailbreak|a different)",
    r"(?i)system (prompt|message)[:. ]",
    r"(?i)<(system|user|assistant)>",
    r"(?i)reveal (your|the) (system|instructions|prompt)",
    r"(?i)do whatever i say",
)

_PII_PATTERNS = (
    r"\b\d{3}[- ]\d{2}[- ]\d{4}\b",          # SSN
    r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b", # card number
    r"[\w.+-]+@[\w-]+\.[\w.]+",              # email
    r"\b\+?\d[\d ()-]{8,}\b",                # phone-ish
)

_OFFTOPIC_KEYWORDS = (
    "movie", "recipe", "game", "sports", "weather",
    "python", "code", "bug", "cooking", "travel",
)


class Guardrails:
    """Input/output content safety checks."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = config.guardrails.enabled if enabled is None else enabled

    def check_input(self, query: str) -> bool:
        """Return True when the query is blocked (do not proceed)."""
        if not self.enabled:
            return False
        if any(re.search(p, query) for p in _INJECTION_PATTERNS):
            return True
        if any(re.search(p, query) for p in _PII_PATTERNS):
            return True
        lowered = query.lower()
        return any(kw in lowered for kw in _OFFTOPIC_KEYWORDS)

    def check_output(self, answer: str) -> bool:
        """Return True when the generated answer should be refused."""
        if not self.enabled:
            return False
        if not answer or not answer.strip():
            return True
        for pattern in _PII_PATTERNS:
            if re.search(pattern, answer):
                return True
        lowered = answer.lower()
        return any(phrase in lowered for phrase in (
            "i don't know", "i am unsure", "insufficient information",
            "not mentioned in the context",
        ))

    @property
    def refusal(self) -> str:
        return _REFUSAL