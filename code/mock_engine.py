"""Deterministic mock engine for unit-testing PCS pipeline without vLLM.

Drop-in for `engine.VLLMEngine`. Implements just enough of the surface for
`portfolio.run_portfolio` and the strategy classes in `strategies_v2.py`
to execute end-to-end on a CPU.

Output content is deterministic per (prompt, seed) so tests can assert
reproducibility.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _MockOutput:
    """Mirrors `engine.GenerationOutput` minimally."""
    text: str
    prompt_tokens: int = 12
    completion_tokens: int = 32
    total_tokens: int = 44
    logprobs: list = field(default_factory=list)
    top_logprobs: list = field(default_factory=list)


class MockEngine:
    """Deterministic mock — no model load, no GPU."""
    def __init__(self, seed: int = 11):
        self.seed = seed
        self.model_name = "mock"

    def _hash_to_text(self, prompt: str, salt: str = "") -> str:
        h = hashlib.sha256((prompt + salt + str(self.seed)).encode()).hexdigest()
        # Pick a digit for "answer" so numeric extraction yields something
        digit = int(h[:2], 16) % 10
        return (
            f"Step 1: Read the problem.\n"
            f"Step 2: Apply formula.\n"
            f"Step 3: Compute carefully.\n"
            f"The answer is {digit}."
        )

    def generate_single(self, prompt: str, max_tokens: int = 512, **kwargs) -> _MockOutput:
        text = self._hash_to_text(prompt)
        return _MockOutput(text=text, completion_tokens=min(40, max_tokens))

    def generate_multi(
        self, prompt: str, n: int = 1, max_tokens: int = 512,
        temperature: float = 0.7, **kwargs,
    ) -> list[_MockOutput]:
        out = []
        for k in range(n):
            text = self._hash_to_text(prompt, salt=f"::sample{k}")
            out.append(_MockOutput(text=text, completion_tokens=min(40, max_tokens)))
        return out

    def generate(self, prompts, **kwargs):
        if isinstance(prompts, str):
            return self.generate_single(prompts, **kwargs)
        return [self.generate_single(p, **kwargs) for p in prompts]

    def apply_chat_template(self, messages):
        return "\n".join(m.get("content", "") for m in messages)

    def get_tokenizer(self):
        return self
