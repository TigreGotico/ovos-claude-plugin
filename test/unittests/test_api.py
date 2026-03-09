"""Unit tests for ovos_claude.api.AnthropicClient."""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_claude.api import AnthropicClient, _get_pronouns, PRONOUN_WORDLISTS


class TestPronounWordlists(unittest.TestCase):
    def test_english_pronouns_present(self):
        pronouns = _get_pronouns("en")
        self.assertIn("it", pronouns)
        self.assertIn("they", pronouns)

    def test_bcp47_tag_fallback(self):
        # "en-us" should fall back to "en" prefix
        pronouns = _get_pronouns("en-us")
        self.assertIn("it", pronouns)

    def test_unknown_lang_falls_back_to_english(self):
        pronouns = _get_pronouns("xx-unknown")
        self.assertEqual(pronouns, PRONOUN_WORDLISTS["en"])

    def test_german_pronouns(self):
        pronouns = _get_pronouns("de")
        self.assertIn("es", pronouns)

    def test_french_pronouns(self):
        pronouns = _get_pronouns("fr")
        self.assertIn("il", pronouns)


class TestAnthropicClient(unittest.TestCase):
    def setUp(self):
        self.config = {
            "api_key": "test-key",
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 1.0,
        }
        self.client = AnthropicClient(self.config)

    def test_model_property(self):
        self.assertEqual(self.client.model, "claude-haiku-4-5-20251001")

    def test_default_model(self):
        c = AnthropicClient({})
        self.assertEqual(c.model, AnthropicClient.DEFAULT_MODEL)

    def test_max_tokens(self):
        self.assertEqual(self.client.max_tokens, 100)

    def test_temperature(self):
        self.assertAlmostEqual(self.client.temperature, 0.5)

    def test_split_system_no_system(self):
        messages = [
            AgentMessage(role=MessageRole.USER, content="Hello"),
            AgentMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ]
        sys_text, conv = AnthropicClient._split_system(messages)
        self.assertIsNone(sys_text)
        self.assertEqual(len(conv), 2)

    def test_split_system_extracts_system(self):
        messages = [
            AgentMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            AgentMessage(role=MessageRole.USER, content="Hello"),
        ]
        sys_text, conv = AnthropicClient._split_system(messages)
        self.assertEqual(sys_text, "You are helpful")
        self.assertEqual(len(conv), 1)
        self.assertEqual(conv[0].role, MessageRole.USER)

    def test_split_system_multiple_system_merged(self):
        messages = [
            AgentMessage(role=MessageRole.SYSTEM, content="Part 1"),
            AgentMessage(role=MessageRole.SYSTEM, content="Part 2"),
            AgentMessage(role=MessageRole.USER, content="Hi"),
        ]
        sys_text, conv = AnthropicClient._split_system(messages)
        self.assertIn("Part 1", sys_text)
        self.assertIn("Part 2", sys_text)
        self.assertEqual(len(conv), 1)

    def test_to_anthropic_messages_roles(self):
        messages = [
            AgentMessage(role=MessageRole.USER, content="Hello"),
            AgentMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ]
        result = AnthropicClient._to_anthropic_messages(messages)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "assistant")

    def test_to_anthropic_messages_content(self):
        messages = [AgentMessage(role=MessageRole.USER, content="test content")]
        result = AnthropicClient._to_anthropic_messages(messages)
        self.assertEqual(result[0]["content"], "test content")

    def test_to_anthropic_messages_multimodal_text_only(self):
        from ovos_plugin_manager.templates.agents import MultimodalAgentMessage
        messages = [
            MultimodalAgentMessage(role=MessageRole.USER, content="What is this?")
        ]
        result = AnthropicClient._to_anthropic_messages_multimodal(messages)
        self.assertEqual(len(result), 1)
        content_blocks = result[0]["content"]
        text_blocks = [b for b in content_blocks if b["type"] == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]["text"], "What is this?")

    def test_to_anthropic_messages_multimodal_with_image(self):
        from ovos_plugin_manager.templates.agents import MultimodalAgentMessage
        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        messages = [
            MultimodalAgentMessage(
                role=MessageRole.USER,
                content="Describe this",
                image_content=[fake_b64],
            )
        ]
        result = AnthropicClient._to_anthropic_messages_multimodal(messages)
        content_blocks = result[0]["content"]
        image_blocks = [b for b in content_blocks if b["type"] == "image"]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]["source"]["type"], "base64")
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/jpeg")

    def test_to_anthropic_messages_multimodal_png_header(self):
        from ovos_plugin_manager.templates.agents import MultimodalAgentMessage
        data_uri = "data:image/png;base64,abc123"
        messages = [
            MultimodalAgentMessage(
                role=MessageRole.USER,
                content="Text",
                image_content=[data_uri],
            )
        ]
        result = AnthropicClient._to_anthropic_messages_multimodal(messages)
        image_blocks = [b for b in result[0]["content"] if b["type"] == "image"]
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/png")
        self.assertEqual(image_blocks[0]["source"]["data"], "abc123")

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_request_calls_messages_create(self, mock_get_client):
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello back")]
        mock_anthropic.messages.create.return_value = mock_response

        messages = [AgentMessage(role=MessageRole.USER, content="Hello")]
        result = self.client.request(messages)

        self.assertEqual(result, "Hello back")
        mock_anthropic.messages.create.assert_called_once()

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_request_passes_system_as_kwarg(self, mock_get_client):
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="reply")]
        mock_anthropic.messages.create.return_value = mock_response

        messages = [
            AgentMessage(role=MessageRole.SYSTEM, content="Be concise"),
            AgentMessage(role=MessageRole.USER, content="Hello"),
        ]
        self.client.request(messages)

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        self.assertIn("system", call_kwargs)
        self.assertEqual(call_kwargs["system"], "Be concise")

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_stream_tokens_yields_text(self, mock_get_client):
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Hello", " ", "world"])
        mock_anthropic.messages.stream.return_value = mock_stream

        messages = [AgentMessage(role=MessageRole.USER, content="Hi")]
        tokens = list(self.client.stream_tokens(messages))
        self.assertEqual(tokens, ["Hello", " ", "world"])


class TestAnthropicClientLazyInit(unittest.TestCase):
    """Test that the anthropic client is lazily initialised."""

    def test_client_is_none_before_first_call(self):
        c = AnthropicClient({"api_key": "test"})
        self.assertIsNone(c._client)

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_client_created_on_first_request(self, mock_get_client):
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_anthropic.messages.create.return_value = mock_response

        c = AnthropicClient({"api_key": "test"})
        c.request([AgentMessage(role=MessageRole.USER, content="hi")])
        mock_get_client.assert_called_once()


if __name__ == "__main__":
    unittest.main()
