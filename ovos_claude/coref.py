"""
ClaudeCoreferenceEngine — opm.agents.coref plugin.

Uses Claude to resolve pronouns and ambiguous references in short
voice-command utterances for multi-turn conversations.

Example::

    context: "Play Bohemian Rhapsody"
    utterance: "Turn it off"
    resolved: "Turn Bohemian Rhapsody off"
"""
from typing import Any, Dict, Optional

from ovos_plugin_manager.templates.agents import (
    AgentMessage,
    CoreferenceEngine,
    MessageRole,
)
from ovos_utils.log import LOG

from ovos_claude.api import AnthropicClient, _get_pronouns


class ClaudeCoreferenceEngine(CoreferenceEngine):
    """
    OVOS CoreferenceEngine backed by Claude.

    Inherits context storage and TTL management from ``CoreferenceEngine``.
    Provides the NLP intelligence via Claude.

    Configuration keys:
        api_key, model, max_tokens (see AnthropicClient).
        system_prompt (str): Override the resolution instruction.
        context_ttl (int):   Seconds to keep context entries (default 120).

    Entry point: ``opm.agents.coref``
    """

    DEFAULT_SYSTEM = (
        "You are a coreference resolver for a voice assistant. "
        "The user will provide a short utterance that may contain pronouns "
        "or ambiguous references (it, they, he, she, this, that, etc.). "
        "Rewrite the utterance with all pronouns and references replaced by "
        "their most likely referents based on context. "
        "Return ONLY the rewritten utterance — no explanation, no quotes."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: str = (
            self.config.get("system_prompt") or self.DEFAULT_SYSTEM
        )

    def contains_corefs(self, text: str, lang: str) -> bool:
        """
        Fast heuristic: return True if *text* contains any known pronoun
        for *lang*.

        This avoids calling the API on utterances that need no resolution.

        Args:
            text: Utterance to check.
            lang: BCP-47 language code.

        Returns:
            True if at least one pronoun is found.
        """
        words = set(text.lower().split())
        pronouns = _get_pronouns(lang)
        return bool(words & set(pronouns))

    def solve_corefs(self, text: str, lang: str) -> str:
        """
        Resolve coreferences in *text* using Claude.

        Args:
            text: Utterance with possible unresolved references.
            lang: BCP-47 language code.

        Returns:
            Rewritten utterance with references resolved.
        """
        try:
            result = self.api.request([
                AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                AgentMessage(role=MessageRole.USER, content=text),
            ])
            return result.strip()
        except Exception as exc:
            LOG.warning(f"ClaudeCoreferenceEngine failed for '{text}': {exc}")
            return text  # Return original on error
