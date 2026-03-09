Last Edit: Claude Sonnet 4.6 - 2026-03-09 - Motive: Initial FAQ for ovos-claude-plugin

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
