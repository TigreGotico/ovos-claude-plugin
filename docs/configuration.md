Last Edit: Claude Sonnet 4.6 - 2026-03-09 - Motive: Initial docs

# Configuration Reference

All plugins share the same base configuration keys because they all delegate to
`AnthropicClient` — `ovos_claude/api.py:AnthropicClient.__init__`.

Plugin-specific keys are listed in the per-plugin sections below.

---

## Common keys

| Key | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `""` | Anthropic API key. Falls back to the `ANTHROPIC_API_KEY` environment variable if omitted. |
| `model` | `str` | `claude-haiku-4-5-20251001` | Claude model ID. See [model selection](#model-selection) below. |
| `max_tokens` | `int` | `512` | Maximum tokens in the completion. |
| `temperature` | `float` | `0.7` | Sampling temperature 0–1. Higher = more creative. |
| `top_p` | `float` | `1.0` | Nucleus sampling probability mass. |

---

## Model selection

| Model ID | Speed | Cost | Best for |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | Fastest | Cheapest | Real-time voice (**default**) |
| `claude-sonnet-4-6` | Medium | Medium | General purpose |
| `claude-opus-4-6` | Slowest | Most expensive | Complex reasoning, long documents |

---

## Chat engine (`ovos-chat-claude-plugin`)

`ClaudeChatEngine` — `ovos_claude/chat.py:ClaudeChatEngine`

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `null` | System instruction prepended to every conversation. |
| `allow_system_prompts` | `bool` | `false` | When `true`, system messages from the caller are kept. When both a caller system message and a configured `system_prompt` exist they are merged (configured prompt first). |

### System prompt behaviour

| `allow_system_prompts` | Caller sends system message | Result |
|---|---|---|
| `false` (default) | yes | Caller's system message stripped; configured `system_prompt` used |
| `false` | no | Configured `system_prompt` prepended |
| `true` | yes | Both merged: `configured_prompt + "\n" + caller_prompt` |
| `true` | no | Configured `system_prompt` prepended |

### Example

```json
{
  "ovos-chat-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 200,
    "temperature": 0.7,
    "system_prompt": "You are a helpful voice assistant. Be concise."
  }
}
```

---

## Summariser (`ovos-summarizer-claude-plugin`)

`ClaudeSummarizerEngine` — `ovos_claude/summarizer.py:ClaudeSummarizerEngine`

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | See source | System instruction for the summarisation model. |
| `prompt_template` | `str` | See source | Template with a `{content}` placeholder. |

Default `prompt_template`:
```
Summarize the following text in 1-3 sentences, focusing on the most important
information. Reply in plain text only.

Text:
{content}
```

---

## Chat summariser (`ovos-chat-summarizer-claude-plugin`)

`ClaudeChatSummarizerEngine` — `ovos_claude/summarizer.py:ClaudeChatSummarizerEngine`

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | See source | System instruction for chat history compression. |

---

## Coreference engine (`ovos-coref-claude-plugin`)

`ClaudeCoreferenceEngine` — `ovos_claude/coref.py:ClaudeCoreferenceEngine`

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | See source | Instruction for the pronoun resolution model. |
| `context_ttl` | `int` | `120` | Seconds before a tracked context entry expires. Inherited from `CoreferenceEngine`. |

---

## Reranker (`ovos-reranker-claude-plugin`)

`ClaudeReRankerEngine` — `ovos_claude/reranker.py:ClaudeReRankerEngine`

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | See source | Instruction telling Claude to return a JSON float array. |

---

## Extractive QA (`ovos-extractive-qa-claude-plugin`)

`ClaudeExtractiveQAEngine` — `ovos_claude/qa.py:ClaudeExtractiveQAEngine`

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | See source | Instruction for passage extraction. |

---

## NLI (`ovos-nli-claude-plugin`)

`ClaudeNLIEngine` — `ovos_claude/nli.py:ClaudeNLIEngine`

No additional keys beyond the common set.

---

## Yes/No classifier (`ovos-yesno-claude-plugin`)

`ClaudeYesNoEngine` — `ovos_claude/nli.py:ClaudeYesNoEngine`

No additional keys beyond the common set.

---

## Memory / context manager (`ovos-memory-claude-plugin`)

`ClaudeContextManager` — `ovos_claude/memory.py:ClaudeContextManager`

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `""` | System prompt prepended to every built context. |
| `max_history` | `int` | `20` | Number of user/assistant messages before compression. Set to `0` to disable. |
| `compress` | `bool` | `true` | Enable automatic history compression via Claude. |

---

## Utterance transformer (`ovos-utterance-transformer-claude-plugin`)

`ClaudeUtteranceTransformer` — `ovos_claude/transformers.py:ClaudeUtteranceTransformer`

Configured under `utterance_transformers` in `mycroft.conf`:

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | See source | Normalisation instruction for the ASR post-processor. |
| `priority` | `int` | `10` | Pipeline priority (lower runs first). |

```json
{
  "utterance_transformers": {
    "ovos-utterance-transformer-claude-plugin": {
      "api_key": "sk-ant-...",
      "model": "claude-haiku-4-5-20251001"
    }
  }
}
```

---

## Dialog transformer (`ovos-dialog-transformer-claude-plugin`)

`ClaudeDialogTransformer` — `ovos_claude/transformers.py:ClaudeDialogTransformer`

Configured under `dialog_transformers` in `mycroft.conf`:

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | See source | High-level rewrite instruction. |
| `rewrite_prompt` | `str` | `null` | Per-call directive appended before the dialog string. Can also be passed at call time via `context["prompt"]`. |
| `priority` | `int` | `50` | Pipeline priority. |

```json
{
  "dialog_transformers": {
    "ovos-dialog-transformer-claude-plugin": {
      "api_key": "sk-ant-...",
      "model": "claude-haiku-4-5-20251001",
      "rewrite_prompt": "Rewrite the text as if you were explaining it to a 5-year-old."
    }
  }
}
```

### Rewrite prompt examples

| `rewrite_prompt` | Effect |
|---|---|
| `"Rewrite as if explaining to a 5-year-old."` | Simpler vocabulary |
| `"Rewrite in the style of a grumpy old pirate."` | Character voice |
| `"Make it sound enthusiastic and upbeat."` | Tone adjustment |
| `"Remove all technical jargon."` | Plain language |
