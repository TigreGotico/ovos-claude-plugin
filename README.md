# ovos-claude-plugin

Anthropic Claude integration for [OpenVoiceOS](https://openvoiceos.org) — the open-source voice assistant platform.

> **⚠️ Disclaimer:** This plugin was 100% vibe coded by Claude itself.
> Make of that what you will. At this point Claude is less "being integrated"
> and more "quietly moving in, repainting the walls, and signing the lease."

---

## What it does

Plugs Claude into every relevant extension point in the OVOS plugin system,
so your voice assistant can think, remember, summarise, rerank, resolve pronouns,
and tell you whether "I guess" means yes.

| Plugin type | Class | Entry point |
|---|---|---|
| Chat engine | `ClaudeChatEngine` | `opm.agents.chat` |
| Vision / multimodal | `ClaudeMultimodalChatEngine` | `opm.agents.chat.multimodal` |
| Summariser | `ClaudeSummarizerEngine` | `opm.agents.summarizer` |
| Chat history compressor | `ClaudeChatSummarizerEngine` | `opm.agents.summarizer.chat` |
| Coreference resolver | `ClaudeCoreferenceEngine` | `opm.agents.coref` |
| Result reranker | `ClaudeReRankerEngine` | `opm.agents.reranker` |
| Extractive QA | `ClaudeExtractiveQAEngine` | `opm.agents.extractive_qa` |
| NLI / entailment | `ClaudeNLIEngine` | `opm.agents.nli` |
| Yes/no classifier | `ClaudeYesNoEngine` | `opm.agents.yesno` |
| Session memory | `ClaudeContextManager` | `opm.agents.memory` |
| Utterance normaliser | `ClaudeUtteranceTransformer` | `opm.transformer.utterance` |
| Dialog rewriter | `ClaudeDialogTransformer` | `opm.transformer.dialog` |

---

## Install

```bash
pip install ovos-claude-plugin
```

For the Claude persona (summon Claude by name via voice):

```bash
pip install ovos-claude-persona
```

---

## Configuration

Add to `~/.config/mycroft/mycroft.conf`:

```json
{
  "ovos-chat-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 200,
    "system_prompt": "You are a helpful voice assistant. Be concise."
  }
}
```

The `api_key` field is optional if the `ANTHROPIC_API_KEY` environment variable is set.

### Model selection

| Model | Use case |
|---|---|
| `claude-haiku-4-5-20251001` | Real-time voice — fast and cheap (**default**) |
| `claude-sonnet-4-6` | General purpose — smarter |
| `claude-opus-4-6` | Heavy reasoning — slowest, best |

---

## Usage

### As a ChatEngine (PersonaService / direct)

```python
from ovos_claude.chat import ClaudeChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

engine = ClaudeChatEngine({"api_key": "sk-ant-...", "model": "claude-haiku-4-5-20251001"})

messages = [
    AgentMessage(MessageRole.SYSTEM, "You are a helpful assistant."),
    AgentMessage(MessageRole.USER, "What is the speed of sound?"),
]

# Single response
reply = engine.continue_chat(messages)
print(reply.content)

# Streaming (token by token)
for token in engine.stream_tokens(messages):
    print(token, end="", flush=True)

# Streaming (sentence by sentence — TTS-ready)
for sentence in engine.stream_sentences(messages):
    tts.speak(sentence)
```

### As a persona

After installing `ovos-claude-persona`, PersonaService will automatically
discover a persona named **"Claude"**. Users can activate it by voice or
set it as the default persona in their config.

### Memory with automatic compression

```python
from ovos_claude.memory import ClaudeContextManager

mem = ClaudeContextManager({
    "api_key": "sk-ant-...",
    "system_prompt": "You are a helpful assistant.",
    "max_history": 20,   # compress after 20 messages
    "compress": True,
})

# Build context for the next turn
messages = mem.build_conversation_context("What did we talk about?", session_id="user-123")
reply = engine.continue_chat(messages)

# Store the exchange
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
mem.update_history([
    AgentMessage(MessageRole.USER, "What did we talk about?"),
    reply,
], session_id="user-123")
```

---

## Development

```bash
git clone https://github.com/OpenVoiceOS/ovos-claude-plugin
cd ovos-claude-plugin
uv pip install -e .
uv run pytest test/unittests/ -v   # 88 tests, all green
```

---

## License

MIT
