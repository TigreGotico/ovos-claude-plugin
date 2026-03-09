"""Unit tests for ovos_claude.nli (ClaudeNLIEngine, ClaudeYesNoEngine)."""
import unittest
from unittest.mock import patch

from ovos_claude.nli import ClaudeNLIEngine, ClaudeYesNoEngine


class TestClaudeNLIEngine(unittest.TestCase):

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_yes_returns_true(self, mock_request):
        mock_request.return_value = "yes"
        engine = ClaudeNLIEngine({"api_key": "test"})
        self.assertTrue(engine.predict_entailment("It is raining", "The weather is wet"))

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_no_returns_false(self, mock_request):
        mock_request.return_value = "no"
        engine = ClaudeNLIEngine({"api_key": "test"})
        self.assertFalse(engine.predict_entailment("It is sunny", "The weather is wet"))

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_yes_case_insensitive(self, mock_request):
        mock_request.return_value = "Yes, it is entailed."
        engine = ClaudeNLIEngine({"api_key": "test"})
        self.assertTrue(engine.predict_entailment("p", "h"))

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_api_error_returns_false(self, mock_request):
        mock_request.side_effect = RuntimeError("API down")
        engine = ClaudeNLIEngine({"api_key": "test"})
        self.assertFalse(engine.predict_entailment("p", "h"))


class TestClaudeYesNoEngine(unittest.TestCase):

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_yes_response(self, mock_request):
        mock_request.return_value = "yes"
        engine = ClaudeYesNoEngine({"api_key": "test"})
        self.assertTrue(engine.yes_or_no("Do you like music?", "sure"))

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_no_response(self, mock_request):
        mock_request.return_value = "no"
        engine = ClaudeYesNoEngine({"api_key": "test"})
        self.assertFalse(engine.yes_or_no("Do you like broccoli?", "not really"))

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_unknown_returns_none(self, mock_request):
        mock_request.return_value = "unknown"
        engine = ClaudeYesNoEngine({"api_key": "test"})
        result = engine.yes_or_no("Ready?", "what do you mean?")
        self.assertIsNone(result)

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_api_error_returns_none(self, mock_request):
        mock_request.side_effect = RuntimeError("error")
        engine = ClaudeYesNoEngine({"api_key": "test"})
        result = engine.yes_or_no("Q?", "A")
        self.assertIsNone(result)

    @patch("ovos_claude.nli.AnthropicClient.request")
    def test_yes_case_insensitive(self, mock_request):
        mock_request.return_value = "Yes, definitely."
        engine = ClaudeYesNoEngine({"api_key": "test"})
        self.assertTrue(engine.yes_or_no("Q?", "A"))


if __name__ == "__main__":
    unittest.main()
