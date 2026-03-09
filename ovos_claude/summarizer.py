"""
ClaudeSummarizerEngine / ClaudeChatSummarizerEngine
— opm.agents.summarizer / opm.agents.summarizer.chat plugins.

Uses Claude to condense long documents or chat histories into concise
summaries suitable for TTS or memory management.
"""
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import (
    AgentMessage,
    ChatSummarizerEngine,
    MessageRole,
    SummarizerEngine,
)

from ovos_claude.api import AnthropicClient


class ClaudeSummarizerEngine(SummarizerEngine):
    """
    OVOS SummarizerEngine backed by Claude.

    Summarises a plain-text document into 1-3 sentences (by default).

    Configuration keys:
        api_key, model, max_tokens, temperature (see AnthropicClient).
        system_prompt (str): Override the system instruction.
        prompt_template (str): Override the user prompt template.
                               Must contain ``{content}`` placeholder.

    Entry point: ``opm.agents.summarizer``
    """

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. Your task is to summarize text "
        "concisely in plain language, with no markdown or bullet points."
    )
    DEFAULT_TEMPLATE = (
        "Summarize the following text in 1-3 sentences, focusing on the "
        "most important information. Reply in plain text only.\n\n"
        "Text:\n{content}"
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: str = (
            self.config.get("system_prompt") or self.DEFAULT_SYSTEM
        )
        self.prompt_template: str = (
            self.config.get("prompt_template") or self.DEFAULT_TEMPLATE
        )

    def summarize(self, document: str, lang: Optional[str] = None) -> str:
        """
        Summarise *document* into a short plain-text excerpt.

        Args:
            document: The full text to summarise.
            lang:     BCP-47 language hint (informational).

        Returns:
            Summary string.
        """
        prompt = self.prompt_template.format(content=document)
        return self.api.request([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt),
        ])


class ClaudeChatSummarizerEngine(ChatSummarizerEngine):
    """
    OVOS ChatSummarizerEngine backed by Claude.

    Converts a structured chat history into a concise narrative summary.
    Useful for long-term persona memory compression — summarise old turns
    to stay within the model's context window.

    Entry point: ``opm.agents.summarizer.chat``
    """

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. Summarize the conversation below "
        "into a concise paragraph that captures the key topics, decisions, "
        "and context. Write in third person. Use plain text, no markdown."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: str = (
            self.config.get("system_prompt") or self.DEFAULT_SYSTEM
        )

    def summarize(self, messages: List[AgentMessage],
                  lang: Optional[str] = None) -> str:
        """
        Summarise *messages* (a conversation) into a short plain-text recap.

        Args:
            messages: Full conversation history.
            lang:     BCP-47 language hint (informational).

        Returns:
            Summary string.
        """
        # Format the conversation as a plain transcript
        lines = []
        for m in messages:
            role_label = m.role.value.capitalize()
            lines.append(f"{role_label}: {m.content}")
        transcript = "\n".join(lines)

        user_prompt = f"Conversation:\n{transcript}\n\nSummary:"

        return self.api.request([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=user_prompt),
        ])
