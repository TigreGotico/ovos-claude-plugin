"""Unit tests for ovos_claude.summarizer."""
import unittest
from unittest.mock import patch

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_claude.summarizer import ClaudeSummarizerEngine, ClaudeChatSummarizerEngine


class TestClaudeSummarizerEngine(unittest.TestCase):

    @patch("ovos_claude.summarizer.AnthropicClient.request")
    def test_summarize_calls_api(self, mock_request):
        mock_request.return_value = "Short summary."
        engine = ClaudeSummarizerEngine({"api_key": "test"})
        result = engine.summarize("A very long document about many things.")
        self.assertEqual(result, "Short summary.")
        mock_request.assert_called_once()

    @patch("ovos_claude.summarizer.AnthropicClient.request")
    def test_summarize_uses_prompt_template(self, mock_request):
        mock_request.return_value = "Summary"
        engine = ClaudeSummarizerEngine({"api_key": "test"})
        engine.summarize("test doc")
        call_args = mock_request.call_args[0][0]  # messages list
        user_msg = next(m for m in call_args if m.role == MessageRole.USER)
        self.assertIn("test doc", user_msg.content)

    @patch("ovos_claude.summarizer.AnthropicClient.request")
    def test_custom_prompt_template(self, mock_request):
        mock_request.return_value = "Custom summary"
        engine = ClaudeSummarizerEngine({
            "api_key": "test",
            "prompt_template": "TL;DR: {content}",
        })
        engine.summarize("some doc")
        call_args = mock_request.call_args[0][0]
        user_msg = next(m for m in call_args if m.role == MessageRole.USER)
        self.assertIn("TL;DR:", user_msg.content)

    @patch("ovos_claude.summarizer.AnthropicClient.request")
    def test_custom_system_prompt(self, mock_request):
        mock_request.return_value = "Response"
        engine = ClaudeSummarizerEngine({
            "api_key": "test",
            "system_prompt": "Custom system",
        })
        engine.summarize("doc")
        call_args = mock_request.call_args[0][0]
        sys_msg = next(m for m in call_args if m.role == MessageRole.SYSTEM)
        self.assertEqual(sys_msg.content, "Custom system")


class TestClaudeChatSummarizerEngine(unittest.TestCase):

    @patch("ovos_claude.summarizer.AnthropicClient.request")
    def test_summarize_includes_conversation_transcript(self, mock_request):
        mock_request.return_value = "Chat summary"
        engine = ClaudeChatSummarizerEngine({"api_key": "test"})
        messages = [
            AgentMessage(MessageRole.USER, "Hello"),
            AgentMessage(MessageRole.ASSISTANT, "Hi there"),
            AgentMessage(MessageRole.USER, "How are you?"),
        ]
        result = engine.summarize(messages)
        self.assertEqual(result, "Chat summary")

        call_args = mock_request.call_args[0][0]
        user_msg = next(m for m in call_args if m.role == MessageRole.USER)
        self.assertIn("Hello", user_msg.content)
        self.assertIn("Hi there", user_msg.content)

    @patch("ovos_claude.summarizer.AnthropicClient.request")
    def test_summarize_empty_messages(self, mock_request):
        mock_request.return_value = "Empty summary"
        engine = ClaudeChatSummarizerEngine({"api_key": "test"})
        result = engine.summarize([])
        self.assertEqual(result, "Empty summary")


if __name__ == "__main__":
    unittest.main()
