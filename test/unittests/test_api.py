"""Unit tests for ovos_claude.api.AnthropicClient."""
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_claude.api import AnthropicClient, ClaudeCodeClient, _get_pronouns, PRONOUN_WORDLISTS


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


class TestAnthropicClientGetClient(unittest.TestCase):
    """Cover _get_client lazy-init and ImportError paths (lines 80–90)."""

    def test_get_client_creates_and_caches_instance(self):
        mock_anthropic_module = MagicMock()
        mock_instance = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_instance

        c = AnthropicClient({"api_key": "k"})
        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            result1 = c._get_client()
            result2 = c._get_client()

        self.assertIs(result1, mock_instance)
        self.assertIs(result2, mock_instance)
        # Anthropic() constructor called exactly once (caching works)
        mock_anthropic_module.Anthropic.assert_called_once_with(api_key="k")

    def test_get_client_import_error(self):
        c = AnthropicClient({})
        with patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaises(ImportError):
                c._get_client()


class TestAnthropicClientMultimodalMediaTypes(unittest.TestCase):
    """Cover gif / webp data-URI branches (lines 162–165)."""

    def _make_message(self, data_uri: str):
        from ovos_plugin_manager.templates.agents import MultimodalAgentMessage
        return MultimodalAgentMessage(
            role=MessageRole.USER, content="test", image_content=[data_uri]
        )

    def test_gif_media_type(self):
        msg = self._make_message("data:image/gif;base64,R0lGOD")
        result = AnthropicClient._to_anthropic_messages_multimodal([msg])
        image_blocks = [b for b in result[0]["content"] if b["type"] == "image"]
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/gif")

    def test_webp_media_type(self):
        msg = self._make_message("data:image/webp;base64,UklGRg==")
        result = AnthropicClient._to_anthropic_messages_multimodal([msg])
        image_blocks = [b for b in result[0]["content"] if b["type"] == "image"]
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/webp")


class TestAnthropicClientErrors(unittest.TestCase):
    """Cover exception re-raise paths in request / stream_tokens (lines 217–219, 246–248)."""

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_request_reraises_exception(self, mock_get_client):
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_anthropic.messages.create.side_effect = RuntimeError("boom")

        c = AnthropicClient({})
        with self.assertRaises(RuntimeError):
            c.request([AgentMessage(role=MessageRole.USER, content="hi")])

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_stream_tokens_with_system_kwarg(self, mock_get_client):
        """Cover the `kwargs['system'] = system` branch (line 241)."""
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["hi"])
        mock_anthropic.messages.stream.return_value = mock_stream

        c = AnthropicClient({})
        messages = [AgentMessage(role=MessageRole.SYSTEM, content="Be brief"),
                    AgentMessage(role=MessageRole.USER, content="Hello")]
        list(c.stream_tokens(messages))

        call_kwargs = mock_anthropic.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["system"], "Be brief")

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_stream_tokens_reraises_exception(self, mock_get_client):
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["chunk"])
        mock_anthropic.messages.stream.side_effect = ValueError("stream error")

        c = AnthropicClient({})
        with self.assertRaises(ValueError):
            list(c.stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))


class TestAnthropicClientRequestMultimodal(unittest.TestCase):
    """Cover request_multimodal (lines 262–281)."""

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_request_multimodal_returns_text(self, mock_get_client):
        from ovos_plugin_manager.templates.agents import MultimodalAgentMessage
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I see a cat")]
        mock_anthropic.messages.create.return_value = mock_response

        c = AnthropicClient({})
        messages = [MultimodalAgentMessage(role=MessageRole.USER, content="What is this?")]
        result = c.request_multimodal(messages)
        self.assertEqual(result, "I see a cat")

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_request_multimodal_passes_system(self, mock_get_client):
        from ovos_plugin_manager.templates.agents import MultimodalAgentMessage
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="reply")]
        mock_anthropic.messages.create.return_value = mock_response

        c = AnthropicClient({})
        messages = [
            MultimodalAgentMessage(role=MessageRole.SYSTEM, content="Describe images"),
            MultimodalAgentMessage(role=MessageRole.USER, content="What is this?"),
        ]
        c.request_multimodal(messages)
        call_kwargs = mock_anthropic.messages.create.call_args[1]
        self.assertIn("system", call_kwargs)

    @patch("ovos_claude.api.AnthropicClient._get_client")
    def test_request_multimodal_reraises_exception(self, mock_get_client):
        mock_anthropic = MagicMock()
        mock_get_client.return_value = mock_anthropic
        mock_anthropic.messages.create.side_effect = RuntimeError("api down")

        c = AnthropicClient({})
        from ovos_plugin_manager.templates.agents import MultimodalAgentMessage
        with self.assertRaises(RuntimeError):
            c.request_multimodal([MultimodalAgentMessage(role=MessageRole.USER, content="hi")])


class TestClaudeCodeClientBinary(unittest.TestCase):
    """Cover binary property paths (lines 316–326)."""

    def test_binary_uses_explicit_config(self):
        c = ClaudeCodeClient({"claude_binary": "/custom/claude"})
        self.assertEqual(c.binary, "/custom/claude")

    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_binary_auto_discovers_on_path(self, _mock):
        c = ClaudeCodeClient({})
        self.assertEqual(c.binary, "/usr/bin/claude")

    @patch("shutil.which", return_value=None)
    def test_binary_raises_when_not_on_path(self, _mock):
        c = ClaudeCodeClient({})
        with self.assertRaises(FileNotFoundError):
            _ = c.binary


class TestClaudeCodeClientFormatHistory(unittest.TestCase):
    """Cover _format_history edge cases (line 352 — SYSTEM skip; assistant label)."""

    def test_system_messages_skipped(self):
        messages = [
            AgentMessage(role=MessageRole.SYSTEM, content="ignored"),
            AgentMessage(role=MessageRole.USER, content="hello"),
        ]
        result = ClaudeCodeClient._format_history(messages)
        self.assertNotIn("ignored", result)
        self.assertIn("User: hello", result)

    def test_assistant_label(self):
        messages = [
            AgentMessage(role=MessageRole.USER, content="hi"),
            AgentMessage(role=MessageRole.ASSISTANT, content="hello back"),
        ]
        result = ClaudeCodeClient._format_history(messages)
        self.assertIn("Assistant: hello back", result)


class TestClaudeCodeClientBuildCmdSystemFromConfig(unittest.TestCase):
    """Cover _build_cmd using system_prompt from config (line 375)."""

    def test_system_prompt_from_config(self):
        c = ClaudeCodeClient({"claude_binary": "/usr/bin/claude", "system_prompt": "Be concise"})
        cmd = c._build_cmd()  # no explicit system= arg
        self.assertIn("--append-system-prompt", cmd)
        self.assertIn("Be concise", cmd)


class TestClaudeCodeClientRequest(unittest.TestCase):
    """Cover ClaudeCodeClient.request paths (lines 406–413)."""

    def _client(self, **cfg):
        cfg.setdefault("claude_binary", "/usr/bin/claude")
        return ClaudeCodeClient(cfg)

    @patch("subprocess.run")
    def test_request_raises_on_nonzero_rc(self, mock_run):
        """Cover lines 406–409 — non-zero returncode."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="oops")
        c = self._client()
        with self.assertRaises(RuntimeError) as ctx:
            c.request([AgentMessage(role=MessageRole.USER, content="hi")])
        self.assertIn("oops", str(ctx.exception))

    @patch("subprocess.run")
    def test_request_returns_stripped_stdout(self, mock_run):
        """Cover line 410 — successful return."""
        mock_run.return_value = MagicMock(returncode=0, stdout="  reply  ", stderr="")
        c = self._client()
        result = c.request([AgentMessage(role=MessageRole.USER, content="hi")])
        self.assertEqual(result, "reply")

    @patch("subprocess.run")
    def test_request_raises_on_timeout(self, mock_run):
        """Cover lines 412–413 — TimeoutExpired."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        c = self._client()
        with self.assertRaises(RuntimeError) as ctx:
            c.request([AgentMessage(role=MessageRole.USER, content="hi")])
        self.assertIn("timed out", str(ctx.exception))


class TestClaudeCodeClientStreamTokens(unittest.TestCase):
    """Cover stream_tokens edge cases (lines 434, 448–482)."""

    def _client(self, **cfg):
        cfg.setdefault("claude_binary", "/usr/bin/claude")
        return ClaudeCodeClient(cfg)

    def _make_proc(self, stdout_lines, returncode=0, stderr=""):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = iter(stdout_lines)
        proc.stderr = io.StringIO(stderr)
        proc.returncode = returncode
        proc.wait.return_value = returncode
        return proc

    @patch("subprocess.Popen")
    def test_stream_appends_system_prompt(self, mock_popen):
        """Cover --append-system-prompt branch in stream_tokens (line 434)."""
        proc = self._make_proc([json.dumps({"type": "text", "text": "hi"}) + "\n"])
        mock_popen.return_value = proc

        c = self._client(system_prompt="Be terse")
        messages = [AgentMessage(role=MessageRole.USER, content="Hello")]
        tokens = list(c.stream_tokens(messages))

        cmd = mock_popen.call_args[0][0]
        self.assertIn("--append-system-prompt", cmd)
        self.assertIn("Be terse", cmd)

    @patch("subprocess.Popen")
    def test_stream_broken_pipe_handled(self, mock_popen):
        """Cover BrokenPipeError on stdin.write (lines 448–449)."""
        proc = self._make_proc([json.dumps({"type": "text", "text": "ok"}) + "\n"])
        proc.stdin.write.side_effect = BrokenPipeError
        mock_popen.return_value = proc

        c = self._client()
        # Should not raise; BrokenPipeError is swallowed
        tokens = list(c.stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))
        self.assertEqual(tokens, ["ok"])

    @patch("subprocess.Popen")
    def test_stream_skips_empty_lines(self, mock_popen):
        """Cover `continue` for empty lines (line 458)."""
        lines = [
            "\n",
            "  \n",
            json.dumps({"type": "text", "text": "hello"}) + "\n",
        ]
        proc = self._make_proc(lines)
        mock_popen.return_value = proc

        tokens = list(self._client().stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))
        self.assertEqual(tokens, ["hello"])

    @patch("subprocess.Popen")
    def test_stream_skips_invalid_json(self, mock_popen):
        """Cover json.JSONDecodeError continue (lines 461–462)."""
        lines = [
            "not-json\n",
            json.dumps({"type": "text", "text": "world"}) + "\n",
        ]
        proc = self._make_proc(lines)
        mock_popen.return_value = proc

        tokens = list(self._client().stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))
        self.assertEqual(tokens, ["world"])

    @patch("subprocess.Popen")
    def test_stream_raises_on_nonzero_returncode(self, mock_popen):
        """Cover returncode != 0 path (lines 475–477)."""
        proc = self._make_proc([], returncode=1, stderr="something went wrong")
        mock_popen.return_value = proc

        with self.assertRaises(RuntimeError) as ctx:
            list(self._client().stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))
        self.assertIn("stream failed", str(ctx.exception))

    @patch("subprocess.Popen")
    def test_stream_wait_timeout_kills_proc(self, mock_popen):
        """Cover proc.wait TimeoutExpired path (lines 470–473)."""
        proc = self._make_proc([])
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        mock_popen.return_value = proc

        with self.assertRaises(RuntimeError) as ctx:
            list(self._client().stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))
        proc.kill.assert_called()
        self.assertIn("timed out", str(ctx.exception))

    @patch("ovos_claude.api.time")
    @patch("subprocess.Popen")
    def test_stream_deadline_exceeded_kills_proc(self, mock_popen, mock_time):
        """Cover per-line deadline check (lines 453–455)."""
        # deadline = time.monotonic() + timeout → 0 + 1 = 1
        # inside loop: time.monotonic() → 9999 > 1 → kill + raise
        mock_time.monotonic.side_effect = [0, 9999]

        lines = [json.dumps({"type": "text", "text": "late"}) + "\n"]
        proc = self._make_proc(lines)
        mock_popen.return_value = proc

        with self.assertRaises(RuntimeError) as ctx:
            list(self._client(timeout=1).stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))
        proc.kill.assert_called()
        self.assertIn("timed out", str(ctx.exception))

    @patch("subprocess.Popen")
    def test_stream_reraises_unexpected_exception(self, mock_popen):
        """Cover outer except block (lines 480–482)."""
        mock_popen.side_effect = OSError("cannot exec")

        with self.assertRaises(OSError):
            list(self._client().stream_tokens([AgentMessage(role=MessageRole.USER, content="hi")]))


if __name__ == "__main__":
    unittest.main()
