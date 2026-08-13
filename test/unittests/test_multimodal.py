"""Unit tests for ovos_claude.multimodal."""
import unittest
from unittest.mock import patch

from ovos_plugin_manager.templates.agents import MessageRole, MultimodalAgentMessage

from ovos_claude.multimodal import ClaudeMultimodalChatEngine


class TestClaudeMultimodalChatEngine(unittest.TestCase):

    @patch("ovos_claude.multimodal.AnthropicClient.request_multimodal")
    def test_continue_chat_returns_assistant_message(self, mock_request):
        mock_request.return_value = "I see a cat in the image."
        engine = ClaudeMultimodalChatEngine(config={"api_key": "test"})
        messages = [
            MultimodalAgentMessage(role=MessageRole.USER, content="What is in this image?")
        ]
        result = engine.continue_chat(messages)
        self.assertIsInstance(result, MultimodalAgentMessage)
        self.assertEqual(result.role, MessageRole.ASSISTANT)
        self.assertEqual(result.content, "I see a cat in the image.")

    @patch("ovos_claude.multimodal.AnthropicClient.request_multimodal")
    def test_continue_chat_accepts_tools_kwarg(self, mock_request):
        """Base MultimodalChatEngine.continue_chat carries `tools`
        unconditionally; an engine that can't use tools must still accept
        and ignore it."""
        mock_request.return_value = "I see a cat in the image."
        engine = ClaudeMultimodalChatEngine(config={"api_key": "test"})
        messages = [
            MultimodalAgentMessage(role=MessageRole.USER, content="What is in this image?")
        ]

        result_none = engine.continue_chat(messages, tools=None)
        self.assertIsInstance(result_none, MultimodalAgentMessage)

        result_with_tools = engine.continue_chat(
            messages, tools=[{"name": "get_weather", "description": "d", "parameters": {}}]
        )
        self.assertIsInstance(result_with_tools, MultimodalAgentMessage)

    @patch("ovos_claude.multimodal.AnthropicClient.request_multimodal")
    def test_system_prompt_injected(self, mock_request):
        mock_request.return_value = "Response"
        engine = ClaudeMultimodalChatEngine(config={
            "api_key": "test",
            "system_prompt": "You are a helpful vision assistant.",
        })
        messages = [
            MultimodalAgentMessage(role=MessageRole.USER, content="Describe this.")
        ]
        engine.continue_chat(messages)
        # Check that the API was called (system prompt is injected internally)
        mock_request.assert_called_once()
        call_args = mock_request.call_args[0][0]
        self.assertEqual(call_args[0].role, MessageRole.SYSTEM)
        self.assertEqual(call_args[0].content, "You are a helpful vision assistant.")

    @patch("ovos_claude.multimodal.AnthropicClient.request_multimodal")
    def test_system_messages_stripped_by_default(self, mock_request):
        mock_request.return_value = "Response"
        engine = ClaudeMultimodalChatEngine(config={"api_key": "test"})
        messages = [
            MultimodalAgentMessage(role=MessageRole.SYSTEM, content="Be concise."),
            MultimodalAgentMessage(role=MessageRole.USER, content="Hello"),
        ]
        engine.continue_chat(messages)
        call_args = mock_request.call_args[0][0]
        # System message should be stripped when allow_system_prompts is False
        self.assertNotEqual(call_args[0].role, MessageRole.SYSTEM)


if __name__ == "__main__":
    unittest.main()
