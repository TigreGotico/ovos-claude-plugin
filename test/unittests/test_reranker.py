"""Unit tests for ovos_claude.reranker.ClaudeReRankerEngine."""
import unittest
from unittest.mock import patch

from ovos_claude.reranker import ClaudeReRankerEngine


class TestClaudeReRankerEngine(unittest.TestCase):

    @patch("ovos_claude.reranker.AnthropicClient.request")
    def test_rerank_returns_sorted_list(self, mock_request):
        mock_request.return_value = "[0.3, 0.9, 0.6]"
        engine = ClaudeReRankerEngine({"api_key": "test"})
        options = ["option A", "option B", "option C"]
        result = engine.rerank("query", options)
        # Should be sorted descending by score
        scores = [r[0] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # Top result should be option B (score 0.9)
        self.assertEqual(result[0][1], "option B")

    @patch("ovos_claude.reranker.AnthropicClient.request")
    def test_rerank_return_index_true(self, mock_request):
        mock_request.return_value = "[0.1, 0.8, 0.5]"
        engine = ClaudeReRankerEngine({"api_key": "test"})
        result = engine.rerank("query", ["a", "b", "c"], return_index=True)
        # Top should be index 1 (score 0.8)
        self.assertEqual(result[0][1], 1)

    @patch("ovos_claude.reranker.AnthropicClient.request")
    def test_rerank_fallback_on_bad_json(self, mock_request):
        mock_request.return_value = "not valid json"
        engine = ClaudeReRankerEngine({"api_key": "test"})
        result = engine.rerank("query", ["a", "b"])
        # Falls back to equal scores of 0.5
        self.assertEqual(len(result), 2)
        for score, _ in result:
            self.assertAlmostEqual(score, 0.5)

    @patch("ovos_claude.reranker.AnthropicClient.request")
    def test_rerank_empty_options(self, mock_request):
        engine = ClaudeReRankerEngine({"api_key": "test"})
        result = engine.rerank("query", [])
        self.assertEqual(result, [])
        mock_request.assert_not_called()

    @patch("ovos_claude.reranker.AnthropicClient.request")
    def test_select_answer(self, mock_request):
        mock_request.return_value = "[0.1, 0.95, 0.3]"
        engine = ClaudeReRankerEngine({"api_key": "test"})
        best = engine.select_answer("query", ["a", "b", "c"])
        self.assertEqual(best, "b")

    @patch("ovos_claude.reranker.AnthropicClient.request")
    def test_backtick_json_stripped(self, mock_request):
        mock_request.return_value = "```json\n[0.9, 0.1]\n```"
        engine = ClaudeReRankerEngine({"api_key": "test"})
        result = engine.rerank("q", ["x", "y"])
        self.assertAlmostEqual(result[0][0], 0.9)


if __name__ == "__main__":
    unittest.main()
