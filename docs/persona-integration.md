Last Edit: Claude Sonnet 4.6 - 2026-03-09 - Motive: Initial docs

# Persona & OVOS Integration

## Using the pre-built Claude persona

Install the companion package:

```bash
pip install ovos-claude-persona
```

This registers a persona named **"Claude"** via the `opm.plugin.persona` entry point.
`PersonaService` discovers and loads it automatically at startup.

Activate by voice:

```
"Chat with Claude"
"Ask Claude what the meaning of life is"
```

Or set Claude as the default persona in `~/.config/ovos_persona/personas/Claude.json`:

```json
{
  "name": "Claude",
  "handlers": ["ovos-chat-claude-plugin"],
  "ovos-chat-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 200,
    "system_prompt": "You are Claude, a helpful voice assistant. Be concise."
  }
}
```

---

## Creating a custom Claude persona

Store a JSON file in `~/.config/ovos_persona/personas/`.

### Minimal persona

```json
{
  "name": "My Claude",
  "handlers": ["ovos-chat-claude-plugin"],
  "ovos-chat-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-sonnet-4-6",
    "system_prompt": "You are a friendly assistant who gives short factual answers."
  }
}
```

### Persona with memory

Pair `ClaudeChatEngine` with `ClaudeContextManager` for persistent session memory:

```json
{
  "name": "Claude with Memory",
  "memory_module": "ovos-memory-claude-plugin",
  "handlers": ["ovos-chat-claude-plugin"],
  "ovos-chat-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-sonnet-4-6",
    "system_prompt": "You are a helpful assistant."
  },
  "ovos-memory-claude-plugin": {
    "api_key": "sk-ant-...",
    "max_history": 30,
    "compress": true
  }
}
```

### High-capability persona (Opus)

```json
{
  "name": "Claude Opus",
  "handlers": ["ovos-chat-claude-plugin"],
  "ovos-chat-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-opus-4-6",
    "max_tokens": 400,
    "system_prompt": "You are an expert assistant. Reason carefully before answering."
  }
}
```

---

## Dialog transformer in mycroft.conf

Add the transformer to the `dialog_transformers` section of `~/.config/mycroft/mycroft.conf`
to rewrite every skill response before it reaches TTS:

```json
{
  "dialog_transformers": {
    "ovos-dialog-transformer-claude-plugin": {
      "api_key": "sk-ant-...",
      "model": "claude-haiku-4-5-20251001",
      "rewrite_prompt": "Rewrite the text so it sounds natural when spoken aloud. Remove any markdown."
    }
  }
}
```

The transformer runs inside `ovos-audio` before TTS synthesis.
Default pipeline priority is `50` — `transformers.py:ClaudeDialogTransformer.__init__`.
Multiple transformers can be stacked; lower priority number = runs earlier.

---

## Utterance transformer in mycroft.conf

Add the utterance normaliser to clean up ASR output before intent matching:

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

Default priority is `10` — `transformers.py:ClaudeUtteranceTransformer.__init__` — so it
runs early in the transformer chain, before other plugins.

---

## Using ClaudeReRankerEngine in OCP

Configure the OCP (OVOS Common Play) media pipeline to use Claude for reranking
search results from multiple media providers:

```json
{
  "reranker": "ovos-reranker-claude-plugin",
  "ovos-reranker-claude-plugin": {
    "api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001"
  }
}
```

---

## Programmatic persona construction

```python
from ovos_claude.chat import ClaudeChatEngine
from ovos_claude.memory import ClaudeContextManager
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

config = {"api_key": "sk-ant-...", "model": "claude-haiku-4-5-20251001"}

engine = ClaudeChatEngine(config)
memory = ClaudeContextManager({**config, "system_prompt": "You are helpful.", "max_history": 20})

session = "my-session"

def chat(utterance: str) -> str:
    ctx = memory.build_conversation_context(utterance, session)
    reply = engine.continue_chat(ctx)
    memory.update_history([AgentMessage(MessageRole.USER, utterance), reply], session)
    return reply.content

print(chat("What is photosynthesis?"))
print(chat("Can you give me a simpler explanation?"))  # Has access to prior turn
```
