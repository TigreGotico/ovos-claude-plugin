"""
ClaudeMultimodalChatEngine — opm.agents.chat.multimodal plugin.

Extends ClaudeChatEngine with vision support by mapping
MultimodalAgentMessage.image_content (base64-encoded images) to the
Anthropic vision format.
"""
from typing import Any, Dict, Iterable, List, Optional

from ovos_plugin_manager.templates.agents import (
    MessageRole,
    MultimodalAgentMessage,
    MultimodalChatEngine,
)
from ovos_utils.log import LOG

from sentence_stream import SentenceBoundaryDetector

from ovos_claude.api import AnthropicClient


class ClaudeMultimodalChatEngine(MultimodalChatEngine):
    """
    OVOS MultimodalChatEngine backed by the Anthropic Claude vision API.

    Supports base64-encoded images via ``MultimodalAgentMessage.image_content``.
    File content (``file_content``) is included as text if the model supports it.

    Configuration keys: same as :class:`ovos_claude.chat.ClaudeChatEngine`.

    Entry point: ``opm.agents.chat.multimodal``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: Optional[str] = self.config.get("system_prompt")
        self.allow_system: bool = bool(self.config.get("allow_system_prompts", False))

    def _prepare_messages(self, messages: List[MultimodalAgentMessage]) -> List[MultimodalAgentMessage]:
        """Strip/inject system prompt consistent with the text-only engine."""
        if not self.allow_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if self.system_prompt:
            sys_msg = MultimodalAgentMessage(
                role=MessageRole.SYSTEM, content=self.system_prompt
            )
            if messages and messages[0].role == MessageRole.SYSTEM:
                if self.allow_system:
                    merged = self.system_prompt + "\n" + messages[0].content
                    messages[0] = MultimodalAgentMessage(
                        role=MessageRole.SYSTEM, content=merged
                    )
                else:
                    messages[0] = sys_msg
            else:
                messages = [sys_msg] + messages

        return messages

    def continue_chat(self, messages: List[MultimodalAgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> MultimodalAgentMessage:
        """
        Generate a response to a multimodal conversation turn.

        Images in ``MultimodalAgentMessage.image_content`` are sent to the
        Anthropic vision API as base64 image blocks.

        Args:
            messages:   Conversation history with optional images.
            session_id: Session identifier (informational).
            lang:       BCP-47 language code hint (informational).
            units:      Preferred unit system (informational).

        Returns:
            MultimodalAgentMessage with the assistant's text reply.
        """
        messages = self._prepare_messages(messages)
        text = self.api.request_multimodal(messages)
        return MultimodalAgentMessage(role=MessageRole.ASSISTANT, content=text)
