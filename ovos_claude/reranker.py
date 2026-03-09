"""
ClaudeReRankerEngine — opm.agents.reranker plugin.

Uses Claude to semantically rank a list of candidate answers / search
results against a query, enabling smarter result selection in the OCP
media pipeline and multi-solver persona setups.
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ovos_plugin_manager.templates.agents import (
    AgentMessage,
    MessageRole,
    ReRankerEngine,
)
from ovos_utils.log import LOG

from ovos_claude.api import AnthropicClient


class ClaudeReRankerEngine(ReRankerEngine):
    """
    OVOS ReRankerEngine backed by Claude.

    Prompts Claude to assign a relevance score (0.0–1.0) to each candidate
    option and returns the list sorted by descending score.

    Configuration keys:
        api_key, model, max_tokens (see AnthropicClient).
        system_prompt (str): Override the ranking instruction.

    Entry point: ``opm.agents.reranker``
    """

    DEFAULT_SYSTEM = (
        "You are a relevance ranking assistant. "
        "The user will give you a query and a numbered list of options. "
        "Score each option from 0.0 (completely irrelevant) to 1.0 "
        "(perfectly relevant) based on how well it answers the query. "
        "Respond with ONLY a JSON array of floats in the same order as "
        "the options — no explanation, no extra text.\n"
        'Example: [0.9, 0.2, 0.7]'
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api = AnthropicClient(self.config)
        self.system_prompt: str = (
            self.config.get("system_prompt") or self.DEFAULT_SYSTEM
        )

    def rerank(self, query: str, options: List[str],
               lang: Optional[str] = None,
               return_index: bool = False) -> List[Tuple[float, Union[str, int]]]:
        """
        Score and sort *options* by relevance to *query* using Claude.

        Args:
            query:        The search / selection query.
            options:      Candidate strings to rank.
            lang:         BCP-47 language hint (informational).
            return_index: If True, return option index instead of text.

        Returns:
            List of (score, option_or_index) tuples, sorted descending.
        """
        if not options:
            return []

        numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        user_prompt = f"Query: {query}\n\nOptions:\n{numbered}"

        scores: List[float] = []
        try:
            raw = self.api.request([
                AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                AgentMessage(role=MessageRole.USER, content=user_prompt),
            ])
            # Extract JSON array even if Claude wraps it in backticks
            raw = raw.strip()
            if raw.startswith("```"):
                # Strip opening fence (may include language hint e.g. ```json)
                raw = raw.lstrip("`")
                # Strip optional language hint on same line as opening fence
                first_newline = raw.find("\n")
                if first_newline != -1:
                    raw = raw[first_newline:].strip()
                # Strip closing fence
                raw = raw.rstrip("`").strip()
            scores = json.loads(raw)
            if not isinstance(scores, list) or len(scores) != len(options):
                raise ValueError(f"Unexpected scores shape: {scores!r}")
            scores = [float(s) for s in scores]
        except Exception as exc:
            LOG.warning(f"ClaudeReRankerEngine failed: {exc}. Using equal scores.")
            scores = [0.5] * len(options)

        pairs = [
            (score, idx if return_index else option)
            for idx, (option, score) in enumerate(zip(options, scores))
        ]
        return sorted(pairs, key=lambda x: x[0], reverse=True)
