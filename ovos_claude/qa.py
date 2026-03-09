"""
ClaudeExtractiveQAEngine — opm.agents.extractive_qa plugin.

Uses Claude to locate the most relevant passage in a source document
that answers a given question.  Used by Wikipedia / document solvers to
present a precise spoken answer before full TTS synthesis.
"""
from typing import Any, Dict, Optional

from ovos_plugin_manager.templates.agents import (
    AgentMessage,
    ExtractiveQAEngine,
    MessageRole,
)
from ovos_utils.log import LOG

from ovos_claude.api import AnthropicClient


class ClaudeExtractiveQAEngine(ExtractiveQAEngine):
    """
    OVOS ExtractiveQAEngine backed by Claude.

    Given an evidence paragraph and a question, Claude identifies and
    returns the specific sentence(s) that contain the answer.

    Configuration keys:
        api_key, model, max_tokens (see AnthropicClient).
        system_prompt (str): Override the extraction instruction.

    Entry point: ``opm.agents.extractive_qa``
    """

    DEFAULT_SYSTEM = (
        "You are a reading comprehension assistant. "
        "The user will give you a passage of text and a question. "
        "Find and quote the exact sentence(s) from the passage that best "
        "answer the question. "
        "Return ONLY the quoted passage — no preamble, no explanation."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: str = (
            self.config.get("system_prompt") or self.DEFAULT_SYSTEM
        )

    def get_best_passage(self, evidence: str, question: str,
                         lang: Optional[str] = None) -> str:
        """
        Extract the passage from *evidence* that best answers *question*.

        Args:
            evidence: Source document text.
            question: The user's question.
            lang:     BCP-47 language hint (informational).

        Returns:
            Extracted passage string.
        """
        user_prompt = (
            f"Passage:\n{evidence}\n\n"
            f"Question: {question}\n\n"
            "Quote the exact sentence(s) from the passage that answer the question:"
        )
        try:
            result = self.api.request([
                AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                AgentMessage(role=MessageRole.USER, content=user_prompt),
            ])
            return result.strip()
        except Exception as exc:
            LOG.warning(f"ClaudeExtractiveQAEngine failed: {exc}")
            return ""
