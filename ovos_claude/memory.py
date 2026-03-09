"""
ClaudeContextManager — opm.agents.memory plugin.

Manages per-session conversation history and builds augmented context
for persona turns.  Optionally uses Claude to summarise old history so
that very long sessions stay within the model's context window.
"""
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import (
    AgentContextManager,
    AgentMessage,
    MessageRole,
)
from ovos_utils.log import LOG

from ovos_claude.api import AnthropicClient


class ClaudeContextManager(AgentContextManager):
    """
    OVOS AgentContextManager with optional Claude-powered memory compression.

    Maintains an in-memory rolling history per session.  When the history
    exceeds *max_history* turns, the oldest half is compressed into a summary
    message using Claude and stored as a single SYSTEM message.

    Configuration keys:
        api_key, model, max_tokens (see AnthropicClient).
        system_prompt (str):  Base system prompt prepended to every context.
        max_history (int):    Max user/assistant messages before compression
                              (default: 20).  Set to 0 to disable compression.
        compress (bool):      Enable automatic history compression (default True).

    Entry point: ``opm.agents.memory``
    """

    _COMPRESS_SYSTEM = (
        "You are a conversation memory manager. "
        "The user will provide a transcript of a past conversation. "
        "Write a concise summary in third-person that captures the key "
        "topics, user preferences, and decisions made. "
        "Use plain text, no markdown."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self._history: Dict[str, List[AgentMessage]] = {}
        self._max_history: int = int(self.config.get("max_history", 20))
        self._compress: bool = bool(self.config.get("compress", True))

    # ------------------------------------------------------------------
    # AgentContextManager interface
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[AgentMessage]:
        """Return the stored message history for *session_id*."""
        return list(self._history.get(session_id, []))

    def update_history(self, new_messages: List[AgentMessage],
                       session_id: str):
        """
        Append *new_messages* to the session history.

        If the history grows beyond *max_history* and compression is enabled,
        the oldest half is summarised and replaced with a single SYSTEM message.

        Args:
            new_messages: Messages to append (typically the latest user + assistant turn).
            session_id:   Session identifier.
        """
        if session_id not in self._history:
            self._history[session_id] = []

        self._history[session_id].extend(new_messages)

        if self._compress and self._max_history > 0:
            conv_msgs = [
                m for m in self._history[session_id]
                if m.role != MessageRole.SYSTEM
            ]
            if len(conv_msgs) > self._max_history:
                self._compress_history(session_id)

    def build_conversation_context(self, utterance: str,
                                   session_id: str) -> List[AgentMessage]:
        """
        Return the augmented message list for the next agent turn.

        Structure:
        1. SYSTEM message with the base system prompt (if configured).
        2. Prior conversation history for this session.
        3. USER message with the current *utterance*.

        Args:
            utterance:  The latest user input.
            session_id: Session identifier.

        Returns:
            List of AgentMessage ready to pass to a ChatEngine.
        """
        messages: List[AgentMessage] = []

        if self.system_prompt:
            messages.append(
                AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            )

        messages.extend(self.get_history(session_id))
        messages.append(AgentMessage(role=MessageRole.USER, content=utterance))

        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_history(self, session_id: str):
        """
        Summarise the oldest half of the conversation history using Claude
        and store it as a single SYSTEM summary message.
        """
        history = self._history[session_id]
        # Separate any existing system messages from the conversation
        system_msgs = [m for m in history if m.role == MessageRole.SYSTEM]
        conv_msgs = [m for m in history if m.role != MessageRole.SYSTEM]

        half = len(conv_msgs) // 2
        to_compress = conv_msgs[:half]
        to_keep = conv_msgs[half:]

        lines = [f"{m.role.value.capitalize()}: {m.content}" for m in to_compress]
        transcript = "\n".join(lines)

        try:
            summary = self.api.request([
                AgentMessage(role=MessageRole.SYSTEM, content=self._COMPRESS_SYSTEM),
                AgentMessage(
                    role=MessageRole.USER,
                    content=f"Conversation:\n{transcript}\n\nSummary:",
                ),
            ])
            summary_msg = AgentMessage(
                role=MessageRole.SYSTEM,
                content=f"[Earlier conversation summary]: {summary.strip()}",
            )
            self._history[session_id] = system_msgs + [summary_msg] + to_keep
            LOG.debug(
                f"ClaudeContextManager: compressed {half} messages for session {session_id}"
            )
        except Exception as exc:
            LOG.warning(f"ClaudeContextManager compression failed: {exc}")
            # Keep history uncompressed on error
