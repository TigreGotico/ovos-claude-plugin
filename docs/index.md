
# OVOS Claude Plugin

This is a collection of [OpenVoiceOS](https://openvoiceos.org/) plugins that connect OVOS to the [Anthropic Claude](https://anthropic.com) API.

## Plugins provided

| Entry point | Class | Purpose |
|---|---|---|
| `opm.agents.chat` | `ClaudeChatEngine` | Multi-turn chat for [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) |
| `opm.agents.chat.multimodal` | `ClaudeMultimodalChatEngine` | Vision (base64 images) plus multi-turn chat |
| `opm.agents.summarizer` | `ClaudeSummarizerEngine` | Condense long documents into plain-text summaries |
| `opm.agents.summarizer.chat` | `ClaudeChatSummarizerEngine` | Compress chat history for long-running sessions |
| `opm.agents.coref` | `ClaudeCoreferenceEngine` | Resolve pronouns and ambiguous references in voice commands |
| `opm.agents.reranker` | `ClaudeReRankerEngine` | Rank candidate answers or search results by meaning |
| `opm.agents.extractive_qa` | `ClaudeExtractiveQAEngine` | Extract the exact passage that answers a question |
| `opm.agents.nli` | `ClaudeNLIEngine` | Predict whether a premise entails a hypothesis |
| `opm.agents.yesno` | `ClaudeYesNoEngine` | Classify ambiguous responses ("I guess") as yes, no, or unknown |
| `opm.agents.memory` | `ClaudeContextManager` | Per-session memory with automatic history compression |
| `opm.transformer.utterance` | `ClaudeUtteranceTransformer` | Normalize noisy ASR output before intent matching |
| `opm.transformer.dialog` | `ClaudeDialogTransformer` | Rewrite skill responses before TTS synthesis |

## Requirements

- Python 3.9+
- `ovos-plugin-manager >= 2.2.3a1, < 3.0.0`
- `anthropic >= 0.20.0`
- `sentence-stream`

## Install

```bash
pip install ovos-claude-plugin
```

For the ready-made Claude persona, also install:

```bash
pip install ovos-claude-persona
```

## Quick start

```python
from ovos_claude.chat import ClaudeChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

engine = ClaudeChatEngine({
    "api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001",
    "system_prompt": "You are a concise and helpful voice assistant.",
})

messages = [AgentMessage(role=MessageRole.USER, content="What is the speed of light?")]
reply = engine.continue_chat(messages)
print(reply.content)
```

## Further reading

- [Plugin reference](plugins.md)
- [Configuration reference](configuration.md)
- [Persona & OVOS integration](persona-integration.md)
