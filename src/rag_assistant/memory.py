"""Rolling conversation memory with token-based trimming.

Keeps at most `max_turns` question/answer pairs, and trims oldest turns
first if a rough token estimate exceeds `max_tokens`, so retrieved
context is never displaced from the prompt by conversation history.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

# Rough heuristic (no tokenizer dependency): ~4 characters per token.
_CHARS_PER_TOKEN = 4


@dataclass
class ConversationMemory:
    max_turns: int = 4
    max_tokens: int = 1500
    turns: List[Tuple[str, str]] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        self._trim()

    def _trim(self) -> None:
        while self.turns and (len(self.turns) > self.max_turns or self._token_estimate() > self.max_tokens):
            self.turns.pop(0)

    def _token_estimate(self) -> int:
        total_chars = sum(len(q) + len(a) for q, a in self.turns)
        return total_chars // _CHARS_PER_TOKEN

    def as_text(self) -> str:
        if not self.turns:
            return "(no previous conversation)"
        lines = []
        for question, answer in self.turns:
            lines.append(f"User: {question}")
            lines.append(f"Assistant: {answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns.clear()
