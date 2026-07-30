"""Unit tests for ovos_claude_persona."""
import unittest

from ovos_claude_persona import CLAUDE_PERSONA


class TestClaudePersona(unittest.TestCase):

    def test_has_name(self):
        self.assertEqual(CLAUDE_PERSONA["name"], "Claude")

    def test_has_solvers(self):
        self.assertIn("solvers", CLAUDE_PERSONA)
        self.assertIsInstance(CLAUDE_PERSONA["solvers"], list)
        self.assertTrue(len(CLAUDE_PERSONA["solvers"]) > 0)

    def test_default_solver_is_chat_engine(self):
        self.assertIn("ovos-chat-claude-plugin", CLAUDE_PERSONA["solvers"])

    def test_has_system_prompt(self):
        self.assertIn("system_prompt", CLAUDE_PERSONA)
        self.assertIsInstance(CLAUDE_PERSONA["system_prompt"], str)
        self.assertTrue(len(CLAUDE_PERSONA["system_prompt"]) > 0)

    def test_plugin_config_present(self):
        self.assertIn("ovos-chat-claude-plugin", CLAUDE_PERSONA)
        cfg = CLAUDE_PERSONA["ovos-chat-claude-plugin"]
        self.assertIn("model", cfg)
        self.assertIn("max_tokens", cfg)
