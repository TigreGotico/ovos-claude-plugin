"""Unit tests for ovos_claude.chat.ClaudeChatEngine."""
import unittest
from unittest.mock import MagicMock, patch

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_claude.chat import ClaudeChatEngine


class TestClaudeChatEnginePrepareMessages(unittest.TestCase):
    """Test _prepare_messages system-prompt handling."""

    def _engine(self, **extra):
        cfg = {"api_key": "test", **extra}
        return ClaudeChatEngine(cfg)

    def test_strips_system_by_default(self):
        engine = self._engine()
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Caller system"),
            AgentMessage(MessageRole.USER, "Hello"),
        ]
        result = engine._prepare_messages(messages)
        roles = [m.role for m in result]
        self.assertNotIn(MessageRole.SYSTEM, roles)

    def test_injects_config_system_prompt(self):
        engine = self._engine(system_prompt="Be brief")
        messages = [AgentMessage(MessageRole.USER, "Hi")]
        result = engine._prepare_messages(messages)
        self.assertEqual(result[0].role, MessageRole.SYSTEM)
        self.assertEqual(result[0].content, "Be brief")

    def test_allow_system_keeps_caller_system(self):
        engine = self._engine(allow_system_prompts=True, system_prompt="Config prompt")
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Caller system"),
            AgentMessage(MessageRole.USER, "Hello"),
        ]
        result = engine._prepare_messages(messages)
        self.assertEqual(result[0].role, MessageRole.SYSTEM)
        # Both prompts should be merged
        self.assertIn("Config prompt", result[0].content)
        self.assertIn("Caller system", result[0].content)

    def test_no_system_prompt_no_system_message(self):
        engine = self._engine()  # no system_prompt in config
        messages = [AgentMessage(MessageRole.USER, "Hi")]
        result = engine._prepare_messages(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, MessageRole.USER)

    def test_existing_system_replaced_when_not_allow(self):
        engine = self._engine(system_prompt="New system")
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Old system"),
            AgentMessage(MessageRole.USER, "Hi"),
        ]
        result = engine._prepare_messages(messages)
        # Old system was stripped, then new was injected
        system_msgs = [m for m in result if m.role == MessageRole.SYSTEM]
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0].content, "New system")


class TestClaudeChatEngineContinueChat(unittest.TestCase):

    @patch("ovos_claude.chat.AnthropicClient.request")
    def test_continue_chat_returns_agent_message(self, mock_request):
        mock_request.return_value = "Hello back"
        engine = ClaudeChatEngine({"api_key": "test"})
        messages = [AgentMessage(MessageRole.USER, "Hello")]
        result = engine.continue_chat(messages)
        self.assertIsInstance(result, AgentMessage)
        self.assertEqual(result.role, MessageRole.ASSISTANT)
        self.assertEqual(result.content, "Hello back")

    @patch("ovos_claude.chat.AnthropicClient.request")
    def test_get_response_convenience(self, mock_request):
        mock_request.return_value = "Sure"
        engine = ClaudeChatEngine({"api_key": "test"})
        result = engine.get_response("Are you there?")
        self.assertEqual(result, "Sure")


class TestClaudeChatEngineStreamTokens(unittest.TestCase):

    @patch("ovos_claude.chat.AnthropicClient.stream_tokens")
    def test_stream_tokens_yields_from_api(self, mock_stream):
        mock_stream.return_value = iter(["Hello", " ", "world"])
        engine = ClaudeChatEngine({"api_key": "test"})
        tokens = list(engine.stream_tokens([AgentMessage(MessageRole.USER, "Hi")]))
        self.assertEqual(tokens, ["Hello", " ", "world"])


class TestClaudeChatEngineStreamSentences(unittest.TestCase):

    @patch("ovos_claude.chat.AnthropicClient.stream_tokens")
    def test_stream_sentences_yields_complete_sentence(self, mock_stream):
        # Feed a complete sentence — SentenceBoundaryDetector should yield it
        mock_stream.return_value = iter(["Hello world."])
        engine = ClaudeChatEngine({"api_key": "test"})
        sentences = list(engine.stream_sentences([AgentMessage(MessageRole.USER, "Hi")]))
        self.assertTrue(len(sentences) >= 1)
        full_text = " ".join(sentences)
        self.assertIn("Hello world", full_text)


if __name__ == "__main__":
    unittest.main()
