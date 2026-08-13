"""Unit tests for ClaudeCodeClient and ClaudeCodeChatEngine."""
import unittest
from unittest.mock import MagicMock, patch, call

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_claude.api import ClaudeCodeClient
from ovos_claude.chat import ClaudeCodeChatEngine


# ---------------------------------------------------------------------------
# ClaudeCodeClient
# ---------------------------------------------------------------------------

class TestClaudeCodeClientProperties(unittest.TestCase):

    def test_default_model(self):
        c = ClaudeCodeClient({})
        self.assertEqual(c.model, ClaudeCodeClient.DEFAULT_MODEL)

    def test_custom_model(self):
        c = ClaudeCodeClient({"model": "opus"})
        self.assertEqual(c.model, "opus")

    def test_default_timeout(self):
        c = ClaudeCodeClient({})
        self.assertEqual(c.timeout, 120)

    def test_custom_timeout(self):
        c = ClaudeCodeClient({"timeout": 30})
        self.assertEqual(c.timeout, 30)

    def test_default_tools_empty(self):
        c = ClaudeCodeClient({})
        self.assertEqual(c.tools, "")

    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_binary_from_path(self, _mock):
        c = ClaudeCodeClient({})
        self.assertEqual(c.binary, "/usr/bin/claude")

    def test_binary_from_config(self):
        c = ClaudeCodeClient({"claude_binary": "/opt/claude"})
        self.assertEqual(c.binary, "/opt/claude")

    @patch("shutil.which", return_value=None)
    def test_binary_not_found_raises(self, _mock):
        c = ClaudeCodeClient({})
        with self.assertRaises(FileNotFoundError):
            _ = c.binary


class TestClaudeCodeClientFormatHistory(unittest.TestCase):

    def test_user_and_assistant_messages(self):
        messages = [
            AgentMessage(MessageRole.USER, "Hello"),
            AgentMessage(MessageRole.ASSISTANT, "Hi there"),
            AgentMessage(MessageRole.USER, "How are you?"),
        ]
        text = ClaudeCodeClient._format_history(messages)
        self.assertIn("User: Hello", text)
        self.assertIn("Assistant: Hi there", text)
        self.assertIn("User: How are you?", text)

    def test_system_messages_excluded(self):
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Be helpful"),
            AgentMessage(MessageRole.USER, "Hello"),
        ]
        text = ClaudeCodeClient._format_history(messages)
        self.assertNotIn("Be helpful", text)
        self.assertIn("User: Hello", text)

    def test_extract_system(self):
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Part one"),
            AgentMessage(MessageRole.SYSTEM, "Part two"),
            AgentMessage(MessageRole.USER, "Hi"),
        ]
        sys_text = ClaudeCodeClient._extract_system(messages)
        self.assertIn("Part one", sys_text)
        self.assertIn("Part two", sys_text)

    def test_extract_system_none_when_absent(self):
        messages = [AgentMessage(MessageRole.USER, "Hi")]
        self.assertIsNone(ClaudeCodeClient._extract_system(messages))


class TestClaudeCodeClientBuildCmd(unittest.TestCase):

    def _client(self, **cfg):
        cfg.setdefault("claude_binary", "/usr/bin/claude")
        return ClaudeCodeClient(cfg)

    def test_print_flag_present(self):
        c = self._client()
        cmd = c._build_cmd()
        self.assertIn("--print", cmd)
        self.assertIn("--output-format", cmd)
        self.assertIn("text", cmd)

    def test_model_in_cmd(self):
        c = self._client(model="haiku")
        cmd = c._build_cmd()
        self.assertIn("--model", cmd)
        self.assertIn("haiku", cmd)

    def test_system_prompt_in_cmd(self):
        c = self._client()
        cmd = c._build_cmd(system="Be brief")
        self.assertIn("--append-system-prompt", cmd)
        self.assertIn("Be brief", cmd)

    def test_no_system_prompt_when_absent(self):
        c = self._client()
        cmd = c._build_cmd()
        self.assertNotIn("--append-system-prompt", cmd)

    def test_prompt_not_in_cmd(self):
        """Prompt is passed via stdin, not appended to argv."""
        c = self._client()
        cmd = c._build_cmd()
        self.assertNotIn("My prompt", cmd)


class TestClaudeCodeClientRequest(unittest.TestCase):

    def _client(self, **cfg):
        cfg.setdefault("claude_binary", "/usr/bin/claude")
        return ClaudeCodeClient(cfg)

    @patch("subprocess.run")
    def test_returns_stdout_stripped(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="  Hello world  ", stderr="")
        c = self._client()
        messages = [AgentMessage(MessageRole.USER, "Hi")]
        result = c.request(messages)
        self.assertEqual(result, "Hello world")

    @patch("subprocess.run")
    def test_raises_on_nonzero_rc(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="some error")
        c = self._client()
        with self.assertRaises(RuntimeError):
            c.request([AgentMessage(MessageRole.USER, "Hi")])

    @patch("subprocess.run")
    def test_system_extracted_from_messages(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        c = self._client()
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Be terse"),
            AgentMessage(MessageRole.USER, "What time is it?"),
        ]
        c.request(messages)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--append-system-prompt", cmd)
        self.assertIn("Be terse", cmd)


class TestClaudeCodeClientStreamTokens(unittest.TestCase):

    def _client(self, **cfg):
        cfg.setdefault("claude_binary", "/usr/bin/claude")
        return ClaudeCodeClient(cfg)

    @patch("subprocess.Popen")
    def test_yields_text_events(self, mock_popen):
        import json as _json
        lines = [
            _json.dumps({"type": "text", "text": "Hello"}) + "\n",
            _json.dumps({"type": "text", "text": " world"}) + "\n",
            _json.dumps({"type": "result", "subtype": "success"}) + "\n",
        ]
        proc = MagicMock()
        proc.stdout.__iter__ = MagicMock(return_value=iter(lines))
        proc.wait.return_value = None
        proc.returncode = 0
        mock_popen.return_value = proc

        c = self._client()
        tokens = list(c.stream_tokens([AgentMessage(MessageRole.USER, "Hi")]))
        self.assertEqual(tokens, ["Hello", " world"])

    @patch("subprocess.Popen")
    def test_skips_non_text_events(self, mock_popen):
        import json as _json
        lines = [
            _json.dumps({"type": "assistant", "message": {}}) + "\n",
            _json.dumps({"type": "text", "text": "Only this"}) + "\n",
        ]
        proc = MagicMock()
        proc.stdout.__iter__ = MagicMock(return_value=iter(lines))
        proc.wait.return_value = None
        proc.returncode = 0
        mock_popen.return_value = proc

        c = self._client()
        tokens = list(c.stream_tokens([AgentMessage(MessageRole.USER, "Hi")]))
        self.assertEqual(tokens, ["Only this"])


# ---------------------------------------------------------------------------
# ClaudeCodeChatEngine
# ---------------------------------------------------------------------------

class TestClaudeCodeChatEnginePrepareMessages(unittest.TestCase):

    def _engine(self, **extra):
        cfg = {"claude_binary": "/usr/bin/claude", **extra}
        return ClaudeCodeChatEngine(cfg)

    def test_strips_system_by_default(self):
        engine = self._engine()
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Caller system"),
            AgentMessage(MessageRole.USER, "Hello"),
        ]
        result = engine._prepare_messages(messages)
        self.assertNotIn(MessageRole.SYSTEM, [m.role for m in result])

    def test_injects_config_system_prompt(self):
        engine = self._engine(system_prompt="Be brief")
        messages = [AgentMessage(MessageRole.USER, "Hi")]
        result = engine._prepare_messages(messages)
        self.assertEqual(result[0].role, MessageRole.SYSTEM)
        self.assertEqual(result[0].content, "Be brief")

    def test_allow_system_merges(self):
        engine = self._engine(allow_system_prompts=True, system_prompt="Config")
        messages = [
            AgentMessage(MessageRole.SYSTEM, "Caller"),
            AgentMessage(MessageRole.USER, "Hello"),
        ]
        result = engine._prepare_messages(messages)
        self.assertIn("Config", result[0].content)
        self.assertIn("Caller", result[0].content)


class TestClaudeCodeChatEngineContinueChat(unittest.TestCase):

    @patch("ovos_claude.api.ClaudeCodeClient.request")
    def test_returns_agent_message(self, mock_request):
        mock_request.return_value = "Hello back"
        engine = ClaudeCodeChatEngine({"claude_binary": "/usr/bin/claude"})
        messages = [AgentMessage(MessageRole.USER, "Hello")]
        result = engine.continue_chat(messages)
        self.assertIsInstance(result, AgentMessage)
        self.assertEqual(result.role, MessageRole.ASSISTANT)
        self.assertEqual(result.content, "Hello back")

    @patch("ovos_claude.api.ClaudeCodeClient.request")
    def test_continue_chat_accepts_tools_kwarg(self, mock_request):
        """Base ChatEngine.continue_chat carries `tools` unconditionally;
        an engine that can't use tools must still accept and ignore it."""
        mock_request.return_value = "Hello back"
        engine = ClaudeCodeChatEngine({"claude_binary": "/usr/bin/claude"})
        messages = [AgentMessage(MessageRole.USER, "Hello")]

        result_none = engine.continue_chat(messages, tools=None)
        self.assertIsInstance(result_none, AgentMessage)

        result_with_tools = engine.continue_chat(
            messages, tools=[{"name": "get_weather", "description": "d", "parameters": {}}]
        )
        self.assertIsInstance(result_with_tools, AgentMessage)

    @patch("ovos_claude.api.ClaudeCodeClient.stream_tokens")
    def test_stream_tokens_delegates(self, mock_stream):
        mock_stream.return_value = iter(["tok1", "tok2"])
        engine = ClaudeCodeChatEngine({"claude_binary": "/usr/bin/claude"})
        tokens = list(engine.stream_tokens([AgentMessage(MessageRole.USER, "Hi")]))
        self.assertEqual(tokens, ["tok1", "tok2"])
