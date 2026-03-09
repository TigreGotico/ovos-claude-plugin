"""Unit tests for ovos_claude.transformers."""
import unittest
from unittest.mock import patch

from ovos_claude.transformers import ClaudeUtteranceTransformer, ClaudeDialogTransformer


class TestClaudeUtteranceTransformer(unittest.TestCase):

    @patch("ovos_claude.transformers.AnthropicClient.request")
    def test_transform_normalises_utterance(self, mock_request):
        mock_request.return_value = "What is 2 plus 2?"
        t = ClaudeUtteranceTransformer(config={"api_key": "test"})
        result_utterances, ctx = t.transform(["whats 2 plus 2 ya know"])
        self.assertEqual(result_utterances[0], "What is 2 plus 2?")
        self.assertIsInstance(ctx, dict)

    @patch("ovos_claude.transformers.AnthropicClient.request")
    def test_transform_multiple_utterances(self, mock_request):
        mock_request.side_effect = ["Normalised A", "Normalised B"]
        t = ClaudeUtteranceTransformer(config={"api_key": "test"})
        result, ctx = t.transform(["raw a", "raw b"])
        self.assertEqual(result, ["Normalised A", "Normalised B"])

    @patch("ovos_claude.transformers.AnthropicClient.request")
    def test_transform_falls_back_on_error(self, mock_request):
        mock_request.side_effect = RuntimeError("API down")
        t = ClaudeUtteranceTransformer(config={"api_key": "test"})
        result, ctx = t.transform(["some utterance"])
        # On error, returns original
        self.assertEqual(result, ["some utterance"])

    def test_default_priority(self):
        t = ClaudeUtteranceTransformer(config={"api_key": "test"})
        self.assertEqual(t.priority, 10)

    def test_custom_priority(self):
        t = ClaudeUtteranceTransformer(priority=5, config={"api_key": "test"})
        self.assertEqual(t.priority, 5)


class TestClaudeDialogTransformer(unittest.TestCase):

    @patch("ovos_claude.transformers.AnthropicClient.request")
    def test_transform_with_context_prompt(self, mock_request):
        mock_request.return_value = "Rewritten dialog"
        t = ClaudeDialogTransformer(config={"api_key": "test"})
        result, ctx = t.transform(
            "Original dialog",
            context={"prompt": "Rewrite in pirate speech"},
        )
        self.assertEqual(result, "Rewritten dialog")

    @patch("ovos_claude.transformers.AnthropicClient.request")
    def test_transform_with_config_prompt(self, mock_request):
        mock_request.return_value = "Rewritten"
        t = ClaudeDialogTransformer(config={
            "api_key": "test",
            "rewrite_prompt": "Make it formal",
        })
        result, ctx = t.transform("informal text", context={})
        self.assertEqual(result, "Rewritten")

    def test_no_prompt_returns_original(self):
        t = ClaudeDialogTransformer(config={"api_key": "test"})
        result, ctx = t.transform("Original", context={})
        self.assertEqual(result, "Original")

    @patch("ovos_claude.transformers.AnthropicClient.request")
    def test_transform_falls_back_on_error(self, mock_request):
        mock_request.side_effect = RuntimeError("API error")
        t = ClaudeDialogTransformer(config={"api_key": "test"})
        result, ctx = t.transform(
            "Original",
            context={"prompt": "Transform this"},
        )
        self.assertEqual(result, "Original")

    def test_context_returned_unchanged(self):
        t = ClaudeDialogTransformer(config={"api_key": "test"})
        ctx_in = {"key": "value"}
        _, ctx_out = t.transform("text", context=ctx_in)
        self.assertEqual(ctx_out, ctx_in)


if __name__ == "__main__":
    unittest.main()
