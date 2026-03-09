"""Unit tests for ovos_claude.qa.ClaudeExtractiveQAEngine."""
import unittest
from unittest.mock import patch

from ovos_claude.qa import ClaudeExtractiveQAEngine


class TestClaudeExtractiveQAEngine(unittest.TestCase):

    @patch("ovos_claude.qa.AnthropicClient.request")
    def test_get_best_passage_returns_extracted_text(self, mock_request):
        mock_request.return_value = "The capital of France is Paris."
        engine = ClaudeExtractiveQAEngine({"api_key": "test"})
        result = engine.get_best_passage(
            evidence="France is a country in Europe. The capital of France is Paris.",
            question="What is the capital of France?",
        )
        self.assertEqual(result, "The capital of France is Paris.")

    @patch("ovos_claude.qa.AnthropicClient.request")
    def test_get_best_passage_includes_evidence_in_prompt(self, mock_request):
        mock_request.return_value = "Passage"
        engine = ClaudeExtractiveQAEngine({"api_key": "test"})
        engine.get_best_passage(evidence="Source text here", question="Q?")

        call_args = mock_request.call_args[0][0]
        from ovos_plugin_manager.templates.agents import MessageRole
        user_msg = next(m for m in call_args if m.role == MessageRole.USER)
        self.assertIn("Source text here", user_msg.content)
        self.assertIn("Q?", user_msg.content)

    @patch("ovos_claude.qa.AnthropicClient.request")
    def test_get_best_passage_strips_whitespace(self, mock_request):
        mock_request.return_value = "  result  "
        engine = ClaudeExtractiveQAEngine({"api_key": "test"})
        result = engine.get_best_passage("doc", "q")
        self.assertEqual(result, "result")

    @patch("ovos_claude.qa.AnthropicClient.request")
    def test_get_best_passage_returns_empty_on_error(self, mock_request):
        mock_request.side_effect = RuntimeError("API error")
        engine = ClaudeExtractiveQAEngine({"api_key": "test"})
        result = engine.get_best_passage("doc", "q")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
