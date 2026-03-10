Last Edit: Claude Sonnet 4.6 - 2026-03-10 - Motive: Add Last Edit header

# ovos-claude-plugin FAQ

## What is ovos-claude-plugin?

A collection of OVOS plugin engine implementations backed by the Anthropic Claude API.
It provides plug-and-play Claude support across all OVOS agent engine extension points:
chat, multimodal vision, summarisation, coreference resolution, reranking, extractive QA,
NLI, yes/no classification, utterance normalisation, dialog rewriting, and memory management.

---

## How do I configure the API key?

Add the following to `~/.config/mycroft/mycroft.conf` under the plugin name:

```json
{
  "ovos-chat-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 200
  }
}
```

Alternatively, set the `ANTHROPIC_API_KEY` environment variable and omit `api_key` from config.

---

## Is there a ready-made persona for PersonaService?

Yes.  The `ovos_claude_persona` package (now shipped inside this repo) provides
a `CLAUDE_PERSONA` dict registered under the `opm.plugin.persona` entry point
as `"Claude"`.

Users can activate it by name ("Hey, ask Claude…") or set it as their default
persona.  No extra install is required — it ships with `ovos-claude-plugin`.

To customise it, add an `ovos-chat-claude-plugin` key in
`~/.config/mycroft/mycroft.conf` under `persona` → `solvers`.

Source: `ovos_claude_persona/__init__.py`

---

## Can I use Claude Code CLI instead of an API key?

Yes.  `ClaudeCodeChatEngine` (`ovos-chat-claude-code-plugin`) delegates to the
system `claude` binary (Claude Code CLI) instead of the Anthropic SDK.  No API
key is required; it uses the authenticated Claude Code session on the host.

Configure it in `mycroft.conf`:

```json
{
  "ovos-chat-claude-code-plugin": {
    "model": "sonnet",
    "system_prompt": "You are a helpful voice assistant.",
    "timeout": 60
  }
}
```

Optional keys:

| Key | Default | Description |
|-----|---------|-------------|
| `claude_binary` | auto (PATH) | Explicit path to the `claude` executable |
| `model` | `"sonnet"` | Model alias or full ID passed to `--model` |
| `system_prompt` | — | System prompt injected into every request |
| `timeout` | `120` | Subprocess timeout in seconds |
| `tools` | `""` | Comma-separated CLI tools to allow (empty = chat only) |
| `allow_system_prompts` | `false` | Merge caller-supplied system messages |

---

## What is the difference between the two chat engines?

| Engine | Entry point key | Backend | Requires |
|--------|----------------|---------|---------|
| `ClaudeChatEngine` | `ovos-chat-claude-plugin` | Anthropic SDK (`anthropic` package) | API key |
| `ClaudeCodeChatEngine` | `ovos-chat-claude-code-plugin` | `claude` CLI subprocess | Claude Code installed + authenticated |

Both implement the same `ChatEngine` interface (`continue_chat`, `stream_tokens`,
`stream_sentences`) so they are interchangeable from the OVOS perspective.

---

## How does ClaudeCodeChatEngine handle multi-turn history?

The conversation history is serialised as a plain-text transcript
(`User: …\nAssistant: …`) and passed as the prompt argument to
`claude --print`.  System messages are extracted and forwarded via
`--system-prompt`.  Streaming uses `--output-format stream-json` and
parses `{"type":"text","text":"…"}` events from the CLI output.

`ClaudeCodeClient._format_history` — `ovos_claude/api.py`
`ClaudeCodeClient.stream_tokens` — `ovos_claude/api.py`

---

## Which Claude model should I use?

| Use case | Recommended model |
|---|---|
| Real-time voice (fast, cheap) | `claude-haiku-4-5-20251001` (default) |
| General purpose | `claude-sonnet-4-6` |
| Complex reasoning | `claude-opus-4-6` |

---

## What plugins are registered?

| Entry point | Class | Purpose |
|---|---|---|
| `opm.agents.chat` | `ClaudeChatEngine` | Multi-turn persona chat |
| `opm.agents.chat.multimodal` | `ClaudeMultimodalChatEngine` | Vision + file support |
| `opm.agents.summarizer` | `ClaudeSummarizerEngine` | Document summarisation |
| `opm.agents.summarizer.chat` | `ClaudeChatSummarizerEngine` | Chat history compression |
| `opm.agents.coref` | `ClaudeCoreferenceEngine` | Pronoun resolution |
| `opm.agents.reranker` | `ClaudeReRankerEngine` | Result ranking |
| `opm.agents.extractive_qa` | `ClaudeExtractiveQAEngine` | Passage extraction |
| `opm.agents.nli` | `ClaudeNLIEngine` | Entailment checking |
| `opm.agents.yesno` | `ClaudeYesNoEngine` | Yes/no classification |
| `opm.agents.memory` | `ClaudeContextManager` | Session memory + compression |
| `opm.transformer.utterance` | `ClaudeUtteranceTransformer` | ASR normalisation |
| `opm.transformer.dialog` | `ClaudeDialogTransformer` | Response rewriting for TTS |

---

## How does the memory compression work?

`ClaudeContextManager` keeps a rolling in-memory history per session.
When the history exceeds `max_history` messages (default: 20), the oldest half is sent to Claude
for summarisation and replaced with a single SYSTEM message.
Set `compress: false` or `max_history: 0` to disable.

---

## How do I use Claude as a Persona?

Install the companion package `ovos-claude-persona`:

```bash
pip install ovos-claude-persona
```

Then activate the "Claude" persona via voice or by setting it as default in `ovos-persona`.

---

## How do I verify OPM discovers the plugins?

```python
from ovos_plugin_manager.agents import find_chat_plugins
print(find_chat_plugins())
# {'ovos-chat-claude-plugin': <class 'ovos_claude.chat.ClaudeChatEngine'>}
```

---

## How do I run the unit tests?

```bash
cd ovos-claude-plugin
uv run pytest test/unittests/ -v
```

All 88 tests should pass.

---

## Why does the UtteranceTransformer not transform every utterance?

By design: if the Claude API fails (network error, rate limit, etc.), the transformer
falls back to returning the original utterance unchanged.  This prevents the voice pipeline
from stalling on API errors.

---

## Why does DialogTransformer only rewrite when a prompt is provided?

To avoid spurious rewrites.  The transformer only invokes Claude when:
1. `context["prompt"]` is set by an upstream skill or plugin, OR
2. `rewrite_prompt` is set in the plugin config.

If neither is set, the original dialog is passed to TTS unchanged.

---

## Does ClaudeCoreferenceEngine support non-English languages?

Yes. `contains_corefs()` uses per-language pronoun wordlists for: English, German, French,
Spanish, Portuguese, and Italian. For unsupported languages it falls back to the English list.
`solve_corefs()` calls Claude which supports many languages natively.

---

## What happens if the Anthropic API returns an error?

All engine classes log a WARNING and fail gracefully:
- `ClaudeChatEngine.continue_chat()` — re-raises (let the caller handle)
- `ClaudeUtteranceTransformer` — returns original utterances
- `ClaudeDialogTransformer` — returns original dialog
- `ClaudeCoreferenceEngine.solve_corefs()` — returns original text
- `ClaudeReRankerEngine` — falls back to equal scores (0.5 each)
- `ClaudeExtractiveQAEngine` — returns empty string
- `ClaudeNLIEngine` — returns False
- `ClaudeYesNoEngine` — returns None
- `ClaudeContextManager` — keeps uncompressed history
