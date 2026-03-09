"""
ClaudeUtteranceTransformer / ClaudeDialogTransformer
— opm.transformer.utterance / opm.transformer.dialog plugins.

UtteranceTransformer: Runs after ASR, before intent matching.
  — Uses Claude to normalise informal / noisy ASR output.

DialogTransformer: Runs after skill response generation, before TTS.
  — Uses Claude to rewrite the response in a requested style/persona.
"""
from typing import Any, Dict, List, Optional, Tuple

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_plugin_manager.templates.transformers import (
    DialogTransformer,
    UtteranceTransformer,
)
from ovos_utils.log import LOG

from ovos_claude.api import AnthropicClient


class ClaudeUtteranceTransformer(UtteranceTransformer):
    """
    Pre-intent utterance normalisation using Claude.

    Fixes ASR errors, resolves abbreviations, and converts informal speech
    to standard form before intent matching.

    Example::

        "whats 2 plus 2 ya know" → "What is 2 plus 2?"

    Configuration keys (under ``utterance_transformers`` in ``mycroft.conf``):
        api_key, model, max_tokens (see AnthropicClient).
        system_prompt (str): Override the normalisation instruction.
        priority (int): Pipeline priority; lower = runs earlier (default 10).

    Entry point: ``opm.transformer.utterance``
    """

    DEFAULT_SYSTEM = (
        "You are an ASR post-processor. The user will give you a raw "
        "transcription that may contain errors, slang, or informal speech. "
        "Rewrite it as a clear, grammatically correct English sentence. "
        "Return ONLY the rewritten sentence — no explanation, no quotes."
    )

    def __init__(self, name: str = "ovos-utterance-transformer-claude-plugin",
                 priority: int = 10, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, priority, config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: str = (
            self.config.get("system_prompt") or self.DEFAULT_SYSTEM
        )

    def transform(self, utterances: List[str],
                  context: Optional[Dict[str, Any]] = None
                  ) -> Tuple[List[str], dict]:
        """
        Normalise each utterance using Claude.

        Args:
            utterances: List of ASR hypotheses (first is the most confident).
            context:    Existing message context dict.

        Returns:
            Tuple of (normalised utterances, updated context dict).
        """
        context = context or {}
        transformed = []
        for utterance in utterances:
            try:
                result = self.api.request([
                    AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                    AgentMessage(role=MessageRole.USER, content=utterance),
                ])
                transformed.append(result.strip())
            except Exception as exc:
                LOG.warning(f"ClaudeUtteranceTransformer failed for '{utterance}': {exc}")
                transformed.append(utterance)  # Fall through to original on error

        return transformed, context


class ClaudeDialogTransformer(DialogTransformer):
    """
    Post-generation dialog style transformer using Claude.

    Rewrites the skill's raw text response before it reaches TTS:
    — Removes markdown / bullet points.
    — Applies a persona voice / tone.
    — Adjusts reading level.

    Configuration keys (under ``dialog_transformers`` in ``mycroft.conf``):
        api_key, model, max_tokens (see AnthropicClient).
        system_prompt (str): System instruction for the rewrite.
        rewrite_prompt (str): Default rewrite directive.
                              May be overridden per-call via ``context["prompt"]``.

    Entry point: ``opm.transformer.dialog``
    """

    DEFAULT_SYSTEM = (
        "You are a text post-processor for a voice assistant. "
        "Rewrite the text so it sounds natural when spoken aloud. "
        "Remove all markdown, bullet points, and special characters. "
        "Use short sentences suitable for text-to-speech synthesis. "
        "Return ONLY the rewritten text — no explanation."
    )

    def __init__(self, name: str = "ovos-dialog-transformer-claude-plugin",
                 priority: int = 50, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, priority, config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: str = (
            self.config.get("system_prompt") or self.DEFAULT_SYSTEM
        )
        self.rewrite_prompt: Optional[str] = self.config.get("rewrite_prompt")

    def transform(self, dialog: str,
                  context: Optional[Dict[str, Any]] = None) -> Tuple[str, dict]:
        """
        Rewrite *dialog* using Claude before TTS synthesis.

        The rewrite directive comes from (in priority order):
        1. ``context["prompt"]`` — set by a skill or upstream plugin.
        2. ``self.rewrite_prompt`` — from plugin config.
        3. If neither is set, Claude is **not** called and *dialog* is
           returned unchanged (avoids spurious rewrites).

        Args:
            dialog:  The skill's raw response text.
            context: Message context dict.

        Returns:
            Tuple of (transformed dialog, context).
        """
        context = context or {}
        prompt = context.get("prompt") or self.rewrite_prompt
        if not prompt:
            return dialog, context

        try:
            result = self.api.request([
                AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                AgentMessage(
                    role=MessageRole.USER,
                    content=f"{prompt}:\n\n{dialog}",
                ),
            ])
            return result.strip(), context
        except Exception as exc:
            LOG.warning(f"ClaudeDialogTransformer failed: {exc}")
            return dialog, context
