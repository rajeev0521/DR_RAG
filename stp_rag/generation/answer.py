"""Prompt assembly and Ollama / local LLM generator."""

from __future__ import annotations

import json
from typing import List, Optional
import requests

PROMPT_TEMPLATE = """You are a helpful and precise research assistant. Answer the question directly using ONLY the provided evidence passages. If the evidence does not contain the answer, say "I cannot determine this from the provided context."

EVIDENCE PASSAGES:
{context}

QUESTION:
{question}

ANSWER:"""


class AnswerGenerator:
    """Generates answers using Ollama or local endpoint, tracking exact prompt text."""

    def __init__(
        self,
        model_name: str = "phi3:mini",
        ollama_url: str = "http://localhost:11434/api/generate",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.temperature = temperature

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """Assembles the exact standardized prompt text."""
        formatted_context = "\n\n---\n\n".join(
            f"Passage [{idx + 1}]:\n{ctx.strip()}" for idx, ctx in enumerate(contexts)
        )
        return PROMPT_TEMPLATE.format(context=formatted_context, question=question.strip())

    def generate(self, question: str, contexts: List[str]) -> tuple[str, str]:
        """
        Generates an answer from evidence passages.

        Returns:
            (answer_text, exact_prompt_text)
        """
        prompt = self.build_prompt(question, contexts)

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            }
            resp = requests.post(self.ollama_url, json=payload, timeout=60)
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                return answer, prompt
            else:
                return f"[Ollama Error {resp.status_code}]", prompt
        except Exception as exc:
            # Fallback for testing when Ollama daemon is offline
            return f"[Ollama Offline: {exc}]", prompt

    def health_check(self) -> bool:
        """Verifies that the generation endpoint is responsive and returns non-empty output."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": "Respond with the single word 'OK'.",
                "stream": False,
            }
            resp = requests.post(self.ollama_url, json=payload, timeout=15)
            if resp.status_code == 200:
                ans = resp.json().get("response", "").strip()
                return len(ans) > 0
            return False
        except Exception:
            return False
