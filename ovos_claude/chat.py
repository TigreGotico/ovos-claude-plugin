"""
ClaudeChatEngine — opm.agents.chat plugin.

Wraps the Anthropic messages API as an OVOS ChatEngine so that any
component loading ``opm.agents.chat`` plugins (PersonaService, etc.)
can use Claude for multi-turn voice conversations.

Also provides ClaudeCodeChatEngine, which uses the system ``claude`` CLI
instead of the SDK — useful when no API key is available.
"""
from typing import Any, Dict, Iterable, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole, ToolsArg
from ovos_utils.log import LOG

from sentence_stream import SentenceBoundaryDetector

from ovos_claude.api import AnthropicClient, ClaudeCodeClient


class ClaudeChatEngine(ChatEngine):
    """
    OVOS ChatEngine backed by the Anthropic Claude API.

    Configuration keys (under the plugin entry in ``settings.json``):
        api_key (str):          Anthropic API key.
        model (str):            Claude model ID
                                (default: ``claude-haiku-4-5-20251001``).
        max_tokens (int):       Max response tokens (default: 512).
        temperature (float):    Sampling temperature 0-1 (default: 0.7).
        top_p (float):          Nucleus sampling probability (default: 1.0).
        system_prompt (str):    Default system prompt.
        allow_system_prompts (bool): If True, keep/merge any system messages
                                     already present in *messages*.
                                     If False (default), strip them and use
                                     only ``system_prompt`` from config.

    Entry point: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: Optional[str] = self.config.get("system_prompt")
        self.allow_system: bool = bool(self.config.get("allow_system_prompts", False))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_messages(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        """
        Enforce system-prompt rules before sending to the API.

        1. Strip existing system messages when ``allow_system`` is False.
        2. Inject the configured ``system_prompt`` (prepend or merge).
        """
        if not self.allow_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if self.system_prompt:
            sys_msg = AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            if messages and messages[0].role == MessageRole.SYSTEM:
                if self.allow_system:
                    # Merge caller's system prompt with ours
                    merged = self.system_prompt + "\n" + messages[0].content
                    messages[0] = AgentMessage(role=MessageRole.SYSTEM, content=merged)
                else:
                    messages[0] = sys_msg
            else:
                messages = [sys_msg] + messages

        return messages

    # ------------------------------------------------------------------
    # ChatEngine interface
    # ------------------------------------------------------------------

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: ToolsArg = None) -> AgentMessage:
        """
        Generate a complete response for the given conversation history.

        Args:
            messages:   Full conversation including prior turns.
            session_id: Session identifier (unused by the API, kept for
                        interface compatibility).
            lang:       BCP-47 language code hint (informational).
            units:      Preferred unit system (informational).
            tools:      Accepted for contract conformance with the base
                        ``ChatEngine`` signature; this engine does not
                        support tool calling and ignores it.

        Returns:
            AgentMessage with ``role=ASSISTANT`` and the model's reply.
        """
        messages = self._prepare_messages(messages)
        text = self.api.request(messages)
        return AgentMessage(role=MessageRole.ASSISTANT, content=text)

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        """
        Stream raw tokens from the API as they are generated.

        Not suitable for direct TTS — yields partial word fragments.
        Use :meth:`stream_sentences` for TTS-compatible output.
        """
        messages = self._prepare_messages(messages)
        yield from self.api.stream_tokens(messages)

    def stream_sentences(self, messages: List[AgentMessage],
                         session_id: str = "default",
                         lang: Optional[str] = None,
                         units: Optional[str] = None) -> Iterable[str]:
        """
        Stream complete sentences buffered from the token stream.

        Uses ``SentenceBoundaryDetector`` to accumulate tokens until a
        grammatical sentence boundary is detected, then yields the sentence.
        Suitable for direct TTS synthesis.
        """
        messages = self._prepare_messages(messages)
        boundary_detector = SentenceBoundaryDetector()

        for token in self.api.stream_tokens(messages):
            yield from boundary_detector.add_chunk(token)

        final = boundary_detector.finish()
        if final:
            yield final


class ClaudeCodeChatEngine(ChatEngine):
    """
    OVOS ChatEngine backed by the system ``claude`` CLI (Claude Code).

    Requires Claude Code to be installed and authenticated on the host
    machine.  No Anthropic API key is needed; the CLI uses its own
    session credentials.

    Configuration keys (under the plugin entry in ``settings.json``):
        claude_binary (str):    Path to the ``claude`` executable.
                                Auto-discovered from PATH if omitted.
        model (str):            Model alias or ID (default: ``"sonnet"``).
        system_prompt (str):    Default system prompt.
        timeout (int):          CLI timeout in seconds (default: 120).
        tools (str):            Allowed tools passed to CLI (default: ``""``
                                — no tools, chat-only mode).
        allow_system_prompts (bool): Merge caller system messages.
                                     Default: False.

    Entry point: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = ClaudeCodeClient(self.config)
        self.system_prompt: Optional[str] = self.config.get("system_prompt")
        self.allow_system: bool = bool(self.config.get("allow_system_prompts", False))

    # ------------------------------------------------------------------
    # Internal helpers (identical policy to ClaudeChatEngine)
    # ------------------------------------------------------------------

    def _prepare_messages(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        if not self.allow_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if self.system_prompt:
            sys_msg = AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            if messages and messages[0].role == MessageRole.SYSTEM:
                if self.allow_system:
                    merged = self.system_prompt + "\n" + messages[0].content
                    messages[0] = AgentMessage(role=MessageRole.SYSTEM, content=merged)
                else:
                    messages[0] = sys_msg
            else:
                messages = [sys_msg] + messages

        return messages

    # ------------------------------------------------------------------
    # ChatEngine interface
    # ------------------------------------------------------------------

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: ToolsArg = None) -> AgentMessage:
        # `tools` accepted for contract conformance with the base ChatEngine
        # signature; this engine does not support tool calling and ignores it.
        messages = self._prepare_messages(messages)
        text = self.api.request(messages)
        return AgentMessage(role=MessageRole.ASSISTANT, content=text)

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        messages = self._prepare_messages(messages)
        yield from self.api.stream_tokens(messages)

    def stream_sentences(self, messages: List[AgentMessage],
                         session_id: str = "default",
                         lang: Optional[str] = None,
                         units: Optional[str] = None) -> Iterable[str]:
        messages = self._prepare_messages(messages)
        boundary_detector = SentenceBoundaryDetector()

        for token in self.api.stream_tokens(messages):
            yield from boundary_detector.add_chunk(token)

        final = boundary_detector.finish()
        if final:
            yield final
