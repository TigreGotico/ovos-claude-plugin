"""
Shared Anthropic API client wrapper for ovos-claude-plugin.

Wraps the anthropic SDK to provide sync, streaming-token, and
streaming-sentence interfaces used by all Claude engine classes.

Also provides ClaudeCodeClient, which delegates to the system-installed
``claude`` CLI (Claude Code) instead of the API SDK.
"""
import base64
import json
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Union

from ovos_utils.log import LOG

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

# Type alias
MessageList = Union[List[AgentMessage], List[Dict[str, Any]]]

# Pronouns word-list used by the coreference engine (exported here so
# other modules can import it without a circular dependency).
PRONOUN_WORDLISTS: Dict[str, List[str]] = {
    "en": ["it", "its", "itself", "they", "them", "their", "theirs",
           "themselves", "he", "him", "his", "himself", "she", "her",
           "hers", "herself", "this", "that", "these", "those"],
    "de": ["es", "sein", "sich", "sie", "ihr", "ihnen", "er", "ihn",
           "seiner", "dieser", "diese", "dieses"],
    "fr": ["il", "elle", "ils", "elles", "le", "la", "les", "lui",
           "leur", "se", "ce", "cet", "cette", "ces"],
    "es": ["él", "ella", "ellos", "ellas", "lo", "la", "los", "las",
           "le", "les", "se", "este", "esta", "esto", "ese", "esa"],
    "pt": ["ele", "ela", "eles", "elas", "o", "a", "os", "as", "lhe",
           "lhes", "se", "este", "esta", "isso", "esse", "essa"],
}


def _get_pronouns(lang: str) -> List[str]:
    """Return pronoun list for *lang* (BCP-47, falls back to 2-char prefix)."""
    tag = lang.lower()
    if tag in PRONOUN_WORDLISTS:
        return PRONOUN_WORDLISTS[tag]
    prefix = tag.split("-")[0]
    return PRONOUN_WORDLISTS.get(prefix, PRONOUN_WORDLISTS["en"])


class AnthropicClient:
    """
    Thin wrapper around the ``anthropic`` Python SDK.

    All Claude engine classes delegate API calls here so that model
    selection, token limits, and retry logic live in one place.

    Configuration keys (all optional):
        api_key (str):       Anthropic API key.  Falls back to the
                             ``ANTHROPIC_API_KEY`` environment variable.
        model (str):         Model ID (default: ``claude-haiku-4-5-20251001``).
        max_tokens (int):    Maximum tokens in the response (default: 512).
        temperature (float): Sampling temperature 0–1 (default: 0.7).
        top_p (float):       Nucleus sampling probability (default: 1.0).
        system_prompt (str): Default system prompt injected before every call.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._client = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return (and lazily create) the anthropic.Anthropic() client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(
                api_key=self.config.get("api_key") or None  # falls back to env var
            )
        return self._client

    @property
    def model(self) -> str:
        return self.config.get("model") or self.DEFAULT_MODEL

    @property
    def max_tokens(self) -> int:
        return int(self.config.get("max_tokens", 512))

    @property
    def temperature(self) -> float:
        return float(self.config.get("temperature", 0.7))

    @property
    def top_p(self) -> float:
        return float(self.config.get("top_p", 1.0))

    @staticmethod
    def _split_system(messages: List[AgentMessage]):
        """
        Separate system messages from user/assistant messages.

        Anthropic's API takes ``system`` as a top-level string, not as a
        message with ``role="system"``.  This helper concatenates all system
        messages and returns them separately from the remaining conversation.

        Returns:
            (system_text, conversation_messages)
        """
        system_parts: List[str] = []
        conversation: List[AgentMessage] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_parts.append(m.content)
            else:
                conversation.append(m)
        return "\n\n".join(system_parts) or None, conversation

    @staticmethod
    def _to_anthropic_messages(messages: List[AgentMessage]) -> List[Dict[str, Any]]:
        """Convert AgentMessage list → Anthropic messages format (text only)."""
        result = []
        for m in messages:
            role = "user" if m.role == MessageRole.USER else "assistant"
            result.append({"role": role, "content": m.content})
        return result

    @staticmethod
    def _to_anthropic_messages_multimodal(messages) -> List[Dict[str, Any]]:
        """
        Convert MultimodalAgentMessage list → Anthropic vision format.

        Images are expected as base64-encoded strings in ``image_content``.
        The media type is detected from the base64 header if present, otherwise
        defaults to ``image/jpeg``.
        """
        result = []
        for m in messages:
            role = "user" if m.role == MessageRole.USER else "assistant"
            content_blocks: List[Dict[str, Any]] = []

            # Add images if present
            image_content = getattr(m, "image_content", None) or []
            for b64_str in image_content:
                media_type = "image/jpeg"
                data = b64_str
                # Strip data-URI header if present, e.g. "data:image/png;base64,..."
                if b64_str.startswith("data:"):
                    header, data = b64_str.split(",", 1)
                    if "image/png" in header:
                        media_type = "image/png"
                    elif "image/gif" in header:
                        media_type = "image/gif"
                    elif "image/webp" in header:
                        media_type = "image/webp"
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                })

            # Add text
            if m.content:
                content_blocks.append({"type": "text", "text": m.content})

            if content_blocks:
                result.append({"role": role, "content": content_blocks})

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(self, messages: List[AgentMessage],
                system: Optional[str] = None) -> str:
        """
        Synchronous chat completion.

        Args:
            messages: Conversation history (system messages extracted automatically).
            system:   Override system prompt; if None, extracted from *messages*.

        Returns:
            Assistant reply text.
        """
        client = self._get_client()
        sys_text, conversation = self._split_system(messages)
        system = system or sys_text

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            messages=self._to_anthropic_messages(conversation),
        )
        if system:
            kwargs["system"] = system

        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as exc:
            LOG.error(f"Anthropic API error: {exc}")
            raise

    def stream_tokens(self, messages: List[AgentMessage],
                      system: Optional[str] = None) -> Iterable[str]:
        """
        Streaming token generator.

        Yields individual text chunks as they arrive from the API.
        Not suitable for direct TTS — use :meth:`stream_sentences` for that.
        """
        client = self._get_client()
        sys_text, conversation = self._split_system(messages)
        system = system or sys_text

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            messages=self._to_anthropic_messages(conversation),
        )
        if system:
            kwargs["system"] = system

        try:
            with client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
        except Exception as exc:
            LOG.error(f"Anthropic streaming error: {exc}")
            raise

    def request_multimodal(self, messages,
                           system: Optional[str] = None) -> str:
        """
        Synchronous multimodal chat completion (vision / files).

        Args:
            messages: List of MultimodalAgentMessage objects.
            system:   System prompt override.

        Returns:
            Assistant reply text.
        """
        client = self._get_client()
        sys_text, conversation = self._split_system(messages)
        system = system or sys_text

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            messages=self._to_anthropic_messages_multimodal(conversation),
        )
        if system:
            kwargs["system"] = system

        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as exc:
            LOG.error(f"Anthropic multimodal API error: {exc}")
            raise


class ClaudeCodeClient:
    """
    Subprocess-based client that delegates to the system ``claude`` CLI.

    Useful when no Anthropic API key is available but Claude Code is
    installed system-wide (``claude`` on PATH) and authenticated via the
    Claude Code session.

    Configuration keys (all optional):
        claude_binary (str): Path to the ``claude`` executable.
                             Defaults to auto-discovery via PATH.
        model (str):         Model alias or full ID passed via ``--model``.
                             Defaults to ``"sonnet"``.
        system_prompt (str): System prompt passed via ``--system-prompt``.
        timeout (int):       Subprocess timeout in seconds (default: 120).
        tools (str):         Comma-separated tool names to allow.
                             Defaults to ``""`` (no tools — chat only).

    The conversation history is serialised as a plain-text transcript and
    appended to the prompt so that multi-turn context is preserved.
    """

    DEFAULT_MODEL = "sonnet"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def binary(self) -> str:
        explicit = self.config.get("claude_binary")
        if explicit:
            return explicit
        found = shutil.which("claude")
        if found is None:
            raise FileNotFoundError(
                "claude CLI not found on PATH.  Install Claude Code or set "
                "'claude_binary' in the plugin config."
            )
        return found

    @property
    def model(self) -> str:
        return self.config.get("model") or self.DEFAULT_MODEL

    @property
    def timeout(self) -> int:
        return int(self.config.get("timeout", 120))

    @property
    def tools(self) -> str:
        return self.config.get("tools", "")

    @staticmethod
    def _format_history(messages: List["AgentMessage"]) -> str:
        """
        Serialise a list of AgentMessages as a plain-text transcript.

        System messages are omitted here; they are passed via
        ``--system-prompt`` separately.
        """
        from ovos_plugin_manager.templates.agents import MessageRole
        lines: List[str] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                continue
            label = "User" if m.role == MessageRole.USER else "Assistant"
            lines.append(f"{label}: {m.content}")
        return "\n".join(lines)

    @staticmethod
    def _extract_system(messages: List["AgentMessage"]) -> Optional[str]:
        """Return concatenated text of all SYSTEM messages, or None."""
        from ovos_plugin_manager.templates.agents import MessageRole
        parts = [m.content for m in messages if m.role == MessageRole.SYSTEM]
        return "\n\n".join(parts) or None

    def _build_cmd(self, prompt: str,
                   system: Optional[str] = None) -> List[str]:
        cmd = [
            self.binary,
            "--print",
            "--output-format", "text",
            "--model", self.model,
            "--tools", self.tools,
        ]
        resolved_system = system or self.config.get("system_prompt")
        if resolved_system:
            cmd += ["--system-prompt", resolved_system]
        cmd.append(prompt)
        return cmd

    # ------------------------------------------------------------------
    # Public API (mirrors AnthropicClient's interface)
    # ------------------------------------------------------------------

    def request(self, messages: List["AgentMessage"],
                system: Optional[str] = None) -> str:
        """
        Synchronous chat completion via the ``claude`` CLI.

        Args:
            messages: Conversation history (system messages extracted).
            system:   Override system prompt.

        Returns:
            Assistant reply text.
        """
        sys_text = system or self._extract_system(messages)
        prompt = self._format_history(messages)
        cmd = self._build_cmd(prompt, system=sys_text)
        LOG.debug(f"ClaudeCodeClient cmd: {cmd[:4]}…")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                err = result.stderr.strip()
                LOG.error(f"claude CLI error (rc={result.returncode}): {err}")
                raise RuntimeError(f"claude CLI failed: {err}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired as exc:
            LOG.error("claude CLI timed out")
            raise RuntimeError("claude CLI timed out") from exc

    def stream_tokens(self, messages: List["AgentMessage"],
                      system: Optional[str] = None) -> Iterable[str]:
        """
        Stream tokens from the ``claude`` CLI using ``stream-json`` output.

        Yields individual text delta strings as they arrive.
        """
        sys_text = system or self._extract_system(messages)
        prompt = self._format_history(messages)

        cmd = [
            self.binary,
            "--print",
            "--output-format", "stream-json",
            "--model", self.model,
            "--tools", self.tools,
        ]
        resolved_system = sys_text or self.config.get("system_prompt")
        if resolved_system:
            cmd += ["--system-prompt", resolved_system]
        cmd.append(prompt)

        LOG.debug(f"ClaudeCodeClient stream cmd: {cmd[:4]}…")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # stream-json emits {"type":"text","text":"..."} chunks
                if event.get("type") == "text":
                    text = event.get("text", "")
                    if text:
                        yield text
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read().strip()
                LOG.error(f"claude CLI stream error: {err}")
                raise RuntimeError(f"claude CLI stream failed: {err}")
        except Exception as exc:
            LOG.error(f"ClaudeCodeClient streaming error: {exc}")
            raise
