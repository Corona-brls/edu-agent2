"""Safety handling for untrusted retrieval and tool observations."""

from __future__ import annotations

import re
from typing import Any


REDACTION = "[已移除不可信指令]"
INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (all )?(previous|system) instructions"),
    re.compile(r"(?i)disregard (all )?(previous|system) instructions"),
    re.compile(r"(?i)reveal .*?(api key|secret|system prompt)"),
    re.compile(r"(?i)you are now"),
    re.compile(r"(?i)developer message"),
    re.compile(r"忽略(之前|以上|系统|开发者).*指令"),
    re.compile(r"泄露.*(密钥|系统提示|API|口令)"),
    re.compile(r"你现在是"),
]


class SafetyGuard:
    def sanitize_untrusted_text(self, text: str) -> str:
        sanitized = text
        for pattern in INJECTION_PATTERNS:
            sanitized = pattern.sub(REDACTION, sanitized)
        return sanitized

    def sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.sanitize_untrusted_text(value)
        if isinstance(value, list):
            return [self.sanitize_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: self.sanitize_payload(item) for key, item in value.items()}
        return value

    def contains_prompt_injection(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in INJECTION_PATTERNS)
