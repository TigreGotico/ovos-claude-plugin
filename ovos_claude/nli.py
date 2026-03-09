"""
ClaudeNLIEngine / ClaudeYesNoEngine
— opm.agents.nli / opm.agents.yesno plugins.

NLI: Determines whether a hypothesis is logically entailed by a premise.
YesNo: Classifies ambiguous responses ("sure", "I guess") as yes/no/None.
"""
from typing import Any, Dict, Optional

from ovos_plugin_manager.templates.agents import (
    AgentMessage,
    MessageRole,
    NaturalLanguageInferenceEngine,
    YesNoEngine,
)
from ovos_utils.log import LOG

from ovos_claude.api import AnthropicClient


class ClaudeNLIEngine(NaturalLanguageInferenceEngine):
    """
    OVOS NaturalLanguageInferenceEngine backed by Claude.

    Determines whether *hypothesis* is logically supported by *premise*.

    Useful for skill confirmation logic — verifying that a user's answer
    is consistent with an expected statement.

    Configuration keys:
        api_key, model, max_tokens (see AnthropicClient).

    Entry point: ``opm.agents.nli``
    """

    SYSTEM_PROMPT = (
        "You are a natural language inference classifier. "
        "The user will give you a premise and a hypothesis. "
        "Determine whether the premise logically entails the hypothesis. "
        'Answer with a single word: "yes" if entailed, "no" if not.'
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)

    def predict_entailment(self, premise: str, hypothesis: str,
                           lang: Optional[str] = None) -> bool:
        """
        Return True if *premise* entails *hypothesis*, False otherwise.

        Args:
            premise:    The base statement or context.
            hypothesis: The statement to verify against the premise.
            lang:       BCP-47 language hint (informational).

        Returns:
            bool — True if entailed.
        """
        user_prompt = (
            f'Premise: "{premise}"\n'
            f'Hypothesis: "{hypothesis}"\n\n'
            "Does the premise entail the hypothesis? Answer yes or no."
        )
        try:
            result = self.api.request([
                AgentMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
                AgentMessage(role=MessageRole.USER, content=user_prompt),
            ])
            return result.strip().lower().startswith("yes")
        except Exception as exc:
            LOG.warning(f"ClaudeNLIEngine failed: {exc}")
            return False


class ClaudeYesNoEngine(YesNoEngine):
    """
    OVOS YesNoEngine backed by Claude.

    Classifies ambiguous user responses to yes/no questions.

    Examples::

        "sure" → True
        "I guess" → True
        "not really" → False
        "what do you mean?" → None

    Configuration keys:
        api_key, model, max_tokens (see AnthropicClient).

    Entry point: ``opm.agents.yesno``
    """

    SYSTEM_PROMPT = (
        "You are a yes/no classifier for a voice assistant. "
        "The user will give you a question and a response. "
        "Classify the response as:\n"
        '  "yes" — if the response means yes, agreement, or confirmation\n'
        '  "no"  — if the response means no, disagreement, or refusal\n'
        '  "unknown" — if the response is ambiguous, off-topic, or unclear\n'
        "Return ONLY one of: yes, no, unknown"
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)

    def yes_or_no(self, question: str, response: str,
                  lang: Optional[str] = None) -> Optional[bool]:
        """
        Classify *response* as a yes (True), no (False), or unclear (None).

        Args:
            question: The yes/no question that was asked.
            response: The user's answer to classify.
            lang:     BCP-47 language hint (informational).

        Returns:
            True, False, or None.
        """
        user_prompt = (
            f'Question: "{question}"\n'
            f'Response: "{response}"\n\n'
            "Is the response yes, no, or unknown?"
        )
        try:
            raw = self.api.request([
                AgentMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
                AgentMessage(role=MessageRole.USER, content=user_prompt),
            ])
            cleaned = raw.strip().lower()
            if cleaned.startswith("yes"):
                return True
            if cleaned.startswith("no"):
                return False
            return None
        except Exception as exc:
            LOG.warning(f"ClaudeYesNoEngine failed: {exc}")
            return None
