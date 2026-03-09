"""Unit tests for ovos_claude.memory.ClaudeContextManager."""
import unittest
from unittest.mock import patch

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_claude.memory import ClaudeContextManager


class TestClaudeContextManagerHistory(unittest.TestCase):

    def setUp(self):
        self.manager = ClaudeContextManager({"api_key": "test", "compress": False})

    def test_get_history_empty_for_new_session(self):
        self.assertEqual(self.manager.get_history("new_session"), [])

    def test_update_and_get_history(self):
        msgs = [
            AgentMessage(MessageRole.USER, "Hello"),
            AgentMessage(MessageRole.ASSISTANT, "Hi"),
        ]
        self.manager.update_history(msgs, "sess1")
        history = self.manager.get_history("sess1")
        self.assertEqual(len(history), 2)

    def test_history_is_returned_as_copy(self):
        self.manager.update_history(
            [AgentMessage(MessageRole.USER, "Test")], "sess2"
        )
        h1 = self.manager.get_history("sess2")
        h1.append(AgentMessage(MessageRole.USER, "Injected"))
        h2 = self.manager.get_history("sess2")
        self.assertEqual(len(h2), 1)  # Internal list unaffected

    def test_update_history_appends(self):
        self.manager.update_history([AgentMessage(MessageRole.USER, "A")], "s")
        self.manager.update_history([AgentMessage(MessageRole.ASSISTANT, "B")], "s")
        self.assertEqual(len(self.manager.get_history("s")), 2)

    def test_sessions_are_isolated(self):
        self.manager.update_history([AgentMessage(MessageRole.USER, "S1")], "s1")
        self.manager.update_history([AgentMessage(MessageRole.USER, "S2")], "s2")
        self.assertEqual(len(self.manager.get_history("s1")), 1)
        self.assertEqual(len(self.manager.get_history("s2")), 1)


class TestClaudeContextManagerBuildContext(unittest.TestCase):

    def test_build_context_no_system_no_history(self):
        manager = ClaudeContextManager({"api_key": "test", "compress": False})
        context = manager.build_conversation_context("Hello", "sess")
        self.assertEqual(len(context), 1)
        self.assertEqual(context[-1].role, MessageRole.USER)
        self.assertEqual(context[-1].content, "Hello")

    def test_build_context_with_system_prompt(self):
        manager = ClaudeContextManager({
            "api_key": "test",
            "system_prompt": "Be helpful",
            "compress": False,
        })
        context = manager.build_conversation_context("Hello", "sess")
        self.assertEqual(context[0].role, MessageRole.SYSTEM)
        self.assertEqual(context[0].content, "Be helpful")
        self.assertEqual(context[-1].role, MessageRole.USER)

    def test_build_context_includes_history(self):
        manager = ClaudeContextManager({"api_key": "test", "compress": False})
        manager.update_history([
            AgentMessage(MessageRole.USER, "Prior"),
            AgentMessage(MessageRole.ASSISTANT, "Response"),
        ], "sess")
        context = manager.build_conversation_context("Now", "sess")
        roles = [m.role for m in context]
        self.assertIn(MessageRole.USER, roles)
        self.assertIn(MessageRole.ASSISTANT, roles)
        self.assertEqual(context[-1].content, "Now")


class TestClaudeContextManagerCompression(unittest.TestCase):

    @patch("ovos_claude.memory.AnthropicClient.request")
    def test_compression_triggered_at_max_history(self, mock_request):
        mock_request.return_value = "Summary of old conversation"
        manager = ClaudeContextManager({
            "api_key": "test",
            "max_history": 4,
            "compress": True,
        })
        # Add 5 user/assistant pairs (10 msgs) to trigger compression
        for i in range(5):
            manager.update_history([
                AgentMessage(MessageRole.USER, f"User msg {i}"),
                AgentMessage(MessageRole.ASSISTANT, f"Asst msg {i}"),
            ], "sess")

        history = manager.get_history("sess")
        # After compression, the history should be shorter than 10 msgs
        self.assertLess(len(history), 10)
        # A summary system message should be present
        system_msgs = [m for m in history if m.role == MessageRole.SYSTEM]
        self.assertGreater(len(system_msgs), 0)
        self.assertIn("Summary", system_msgs[0].content)

    @patch("ovos_claude.memory.AnthropicClient.request")
    def test_compression_fallback_on_error(self, mock_request):
        mock_request.side_effect = RuntimeError("API down")
        manager = ClaudeContextManager({
            "api_key": "test",
            "max_history": 2,
            "compress": True,
        })
        for i in range(3):
            manager.update_history([
                AgentMessage(MessageRole.USER, f"U{i}"),
                AgentMessage(MessageRole.ASSISTANT, f"A{i}"),
            ], "sess")

        # Even if compression fails, history should still be retrievable
        history = manager.get_history("sess")
        self.assertGreater(len(history), 0)

    def test_no_compression_when_disabled(self):
        manager = ClaudeContextManager({
            "api_key": "test",
            "max_history": 2,
            "compress": False,
        })
        for i in range(5):
            manager.update_history([
                AgentMessage(MessageRole.USER, f"U{i}"),
            ], "sess")

        history = manager.get_history("sess")
        self.assertEqual(len(history), 5)


if __name__ == "__main__":
    unittest.main()
