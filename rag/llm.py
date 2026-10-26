"""Optional LLM client for any OpenAI-compatible Chat Completions endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from the provided context. "
    "Cite sources inline using their bracketed numbers, e.g. [1]. If the context does "
    "not contain the answer, say you don't know. Answer in the user's language."
)


@dataclass
class LLMConfig:
    api_key: str | None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class LLMClient:
    def __init__(self, config: LLMConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._transport = transport  # injectable for tests

    def complete(self, question: str, context: str) -> str:
        payload = {
            "model": self.config.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        with httpx.Client(timeout=self.config.timeout, transport=self._transport) as client:
            resp = client.post(f"{self.config.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
