"""Unit tests for ovos_claude.coref.ClaudeCoreferenceEngine."""
import unittest
from unittest.mock import patch

from ovos_claude.coref import ClaudeCoreferenceEngine


class TestClaudeCoreferenceEngineContainsCorefs(unittest.TestCase):

    def setUp(self):
        self.engine = ClaudeCoreferenceEngine({"api_key": "test"})

    def test_detects_pronoun_it(self):
        self.assertTrue(self.engine.contains_corefs("turn it off", "en"))

    def test_detects_pronoun_they(self):
        self.assertTrue(self.engine.contains_corefs("can they hear me", "en"))

    def test_no_pronoun_returns_false(self):
        self.assertFalse(self.engine.contains_corefs("play bohemian rhapsody", "en"))

    def test_case_insensitive(self):
        self.assertTrue(self.engine.contains_corefs("Turn IT off", "en"))

    def test_german_pronoun(self):
        self.assertTrue(self.engine.contains_corefs("mach es aus", "de"))

    def test_unknown_lang_falls_back(self):
        # Should not raise; uses English pronouns as fallback
        result = self.engine.contains_corefs("turn it off", "xx")
        self.assertTrue(result)


class TestClaudeCoreferenceEngineSolveCorefs(unittest.TestCase):

    @patch("ovos_claude.coref.AnthropicClient.request")
    def test_solve_returns_resolved_text(self, mock_request):
        mock_request.return_value = "Turn Bohemian Rhapsody off"
        engine = ClaudeCoreferenceEngine({"api_key": "test"})
        result = engine.solve_corefs("Turn it off", "en")
        self.assertEqual(result, "Turn Bohemian Rhapsody off")

    @patch("ovos_claude.coref.AnthropicClient.request")
    def test_solve_returns_original_on_error(self, mock_request):
        mock_request.side_effect = RuntimeError("API error")
        engine = ClaudeCoreferenceEngine({"api_key": "test"})
        result = engine.solve_corefs("Turn it off", "en")
        self.assertEqual(result, "Turn it off")

    @patch("ovos_claude.coref.AnthropicClient.request")
    def test_solve_strips_whitespace(self, mock_request):
        mock_request.return_value = "  resolved text  "
        engine = ClaudeCoreferenceEngine({"api_key": "test"})
        result = engine.solve_corefs("text", "en")
        self.assertEqual(result, "resolved text")


class TestClaudeCoreferenceEngineResolve(unittest.TestCase):
    """Test the inherited resolve() method which calls contains_corefs + solve_corefs."""

    @patch("ovos_claude.coref.AnthropicClient.request")
    def test_resolve_calls_solve_when_pronouns_present(self, mock_request):
        mock_request.return_value = "Resolved"
        engine = ClaudeCoreferenceEngine({"api_key": "test"})
        result = engine.resolve("turn it off", lang="en")
        self.assertEqual(result, "Resolved")
        mock_request.assert_called_once()

    def test_resolve_skips_api_when_no_pronouns(self):
        engine = ClaudeCoreferenceEngine({"api_key": "test"})
        result = engine.resolve("play bohemian rhapsody", lang="en")
        self.assertEqual(result, "play bohemian rhapsody")


if __name__ == "__main__":
    unittest.main()
