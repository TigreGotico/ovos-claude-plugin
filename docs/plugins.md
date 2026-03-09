Last Edit: Claude Sonnet 4.6 - 2026-03-09 - Motive: Initial docs

# Plugin Reference

---

## ovos-chat-claude-plugin

**Class:** `ovos_claude.chat.ClaudeChatEngine`
**OPM group:** `opm.agents.chat`
**Source:** `ovos_claude/chat.py:ClaudeChatEngine`

A `ChatEngine` implementation for multi-turn conversations via the Anthropic Messages API.
Supports synchronous responses, raw token streaming, and sentence-level streaming for TTS.

### Key methods

| Method | Description |
|---|---|
| `continue_chat(messages, session_id, lang, units) → AgentMessage` | Single blocking response — `chat.py:ClaudeChatEngine.continue_chat` |
| `stream_tokens(messages, ...) → Iterable[str]` | Raw token stream, not suitable for direct TTS — `chat.py:ClaudeChatEngine.stream_tokens` |
| `stream_sentences(messages, ...) → Iterable[str]` | Sentence-buffered stream via `SentenceBoundaryDetector`, TTS-ready — `chat.py:ClaudeChatEngine.stream_sentences` |
| `get_response(utterance, ...) → str` | Convenience single-turn wrapper inherited from `ChatEngine` |

### Usage

```python
from ovos_claude.chat import ClaudeChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

engine = ClaudeChatEngine({
    "api_key": "sk-ant-...",
    "model": "claude-haiku-4-5-20251001",
    "system_prompt": "You are a helpful assistant. Be concise.",
})

history = [AgentMessage(role=MessageRole.USER, content="Tell me a joke.")]

# Single blocking response
reply = engine.continue_chat(history)
print(reply.content)

# Stream token by token
for tok in engine.stream_tokens(history):
    print(tok, end="", flush=True)

# Stream sentence by sentence (pipe directly to TTS)
for sentence in engine.stream_sentences(history):
    tts.speak(sentence)
```

---

## ovos-chat-multimodal-claude-plugin

**Class:** `ovos_claude.multimodal.ClaudeMultimodalChatEngine`
**OPM group:** `opm.agents.chat.multimodal`
**Source:** `ovos_claude/multimodal.py:ClaudeMultimodalChatEngine`

Extends the chat engine with vision support.
Images must be passed as base64-encoded strings in `MultimodalAgentMessage.image_content`.
Data-URI headers (`data:image/png;base64,...`) are stripped automatically.

### Usage

```python
from ovos_claude.multimodal import ClaudeMultimodalChatEngine
from ovos_plugin_manager.templates.agents import MultimodalAgentMessage, MessageRole
import base64

engine = ClaudeMultimodalChatEngine({"api_key": "sk-ant-..."})

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

messages = [
    MultimodalAgentMessage(
        role=MessageRole.USER,
        content="What is in this image?",
        image_content=[b64],
    )
]
reply = engine.continue_chat(messages)
print(reply.content)
```

---

## ovos-summarizer-claude-plugin

**Class:** `ovos_claude.summarizer.ClaudeSummarizerEngine`
**OPM group:** `opm.agents.summarizer`
**Source:** `ovos_claude/summarizer.py:ClaudeSummarizerEngine`

Condenses a plain-text document into 1–3 sentences.
Intended for consumption by skills (Wikipedia solver, news reader) before TTS.

### Usage

```python
from ovos_claude.summarizer import ClaudeSummarizerEngine

summarizer = ClaudeSummarizerEngine({"api_key": "sk-ant-..."})
summary = summarizer.summarize(long_article_text)
print(summary)
```

---

## ovos-chat-summarizer-claude-plugin

**Class:** `ovos_claude.summarizer.ClaudeChatSummarizerEngine`
**OPM group:** `opm.agents.summarizer.chat`
**Source:** `ovos_claude/summarizer.py:ClaudeChatSummarizerEngine`

Converts a structured chat history into a concise narrative summary.
Used internally by `ClaudeContextManager` for memory compression; can also be called directly.

### Usage

```python
from ovos_claude.summarizer import ClaudeChatSummarizerEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

summarizer = ClaudeChatSummarizerEngine({"api_key": "sk-ant-..."})

messages = [
    AgentMessage(MessageRole.USER, "What's the weather like?"),
    AgentMessage(MessageRole.ASSISTANT, "It's sunny and 22°C in Lisbon."),
    AgentMessage(MessageRole.USER, "Will it rain tomorrow?"),
    AgentMessage(MessageRole.ASSISTANT, "Light showers are expected in the afternoon."),
]

print(summarizer.summarize(messages))
# "The user asked about the weather in Lisbon. The assistant reported sunny conditions
#  at 22°C and warned of light afternoon showers the following day."
```

---

## ovos-coref-claude-plugin

**Class:** `ovos_claude.coref.ClaudeCoreferenceEngine`
**OPM group:** `opm.agents.coref`
**Source:** `ovos_claude/coref.py:ClaudeCoreferenceEngine`

Resolves pronouns and ambiguous references in voice commands.
`contains_corefs()` uses a fast per-language pronoun wordlist — `api.py:PRONOUN_WORDLISTS` —
to avoid calling the API on utterances that need no resolution.

### Supported languages (wordlist)

English, German, French, Spanish, Portuguese. All other languages fall back to English.
`solve_corefs()` calls Claude which handles many more languages natively.

### Usage

```python
from ovos_claude.coref import ClaudeCoreferenceEngine

engine = ClaudeCoreferenceEngine({"api_key": "sk-ant-..."})

# After "Play Bohemian Rhapsody":
result = engine.resolve("Turn it off", lang="en")
print(result)  # "Turn Bohemian Rhapsody off"
```

---

## ovos-reranker-claude-plugin

**Class:** `ovos_claude.reranker.ClaudeReRankerEngine`
**OPM group:** `opm.agents.reranker`
**Source:** `ovos_claude/reranker.py:ClaudeReRankerEngine`

Prompts Claude to score each candidate 0.0–1.0, then returns the list sorted descending.
Falls back to equal scores (0.5) on API errors or malformed JSON.
JSON wrapped in markdown backtick fences is stripped automatically.

### Usage

```python
from ovos_claude.reranker import ClaudeReRankerEngine

engine = ClaudeReRankerEngine({"api_key": "sk-ant-..."})

options = ["Bohemian Rhapsody by Queen", "Bohemian Like You by Dandy Warhols", "Bohemian Groove Mix"]
ranked = engine.rerank("play bohemian rhapsody", options)
# [(0.97, "Bohemian Rhapsody by Queen"), (0.34, "Bohemian Groove Mix"), (0.12, "Bohemian Like You by Dandy Warhols")]

best = engine.select_answer("play bohemian rhapsody", options)
print(best)  # "Bohemian Rhapsody by Queen"
```

---

## ovos-extractive-qa-claude-plugin

**Class:** `ovos_claude.qa.ClaudeExtractiveQAEngine`
**OPM group:** `opm.agents.extractive_qa`
**Source:** `ovos_claude/qa.py:ClaudeExtractiveQAEngine`

Given an evidence paragraph and a question, Claude identifies and quotes the relevant sentence(s).
Returns an empty string on API error.

### Usage

```python
from ovos_claude.qa import ClaudeExtractiveQAEngine

engine = ClaudeExtractiveQAEngine({"api_key": "sk-ant-..."})

evidence = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. "
    "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. "
    "It stands 330 metres tall."
)
answer = engine.get_best_passage(evidence, "How tall is the Eiffel Tower?")
print(answer)  # "It stands 330 metres tall."
```

---

## ovos-nli-claude-plugin

**Class:** `ovos_claude.nli.ClaudeNLIEngine`
**OPM group:** `opm.agents.nli`
**Source:** `ovos_claude/nli.py:ClaudeNLIEngine`

Determines whether a *premise* logically entails a *hypothesis*.
Returns `False` on API error.

### Usage

```python
from ovos_claude.nli import ClaudeNLIEngine

engine = ClaudeNLIEngine({"api_key": "sk-ant-..."})

print(engine.predict_entailment("It is raining heavily.", "The weather is wet."))  # True
print(engine.predict_entailment("It is sunny.", "You need an umbrella."))          # False
```

---

## ovos-yesno-claude-plugin

**Class:** `ovos_claude.nli.ClaudeYesNoEngine`
**OPM group:** `opm.agents.yesno`
**Source:** `ovos_claude/nli.py:ClaudeYesNoEngine`

Classifies a user's response to a yes/no question as `True`, `False`, or `None` (unclear).
Returns `None` on API error.

### Usage

```python
from ovos_claude.nli import ClaudeYesNoEngine

engine = ClaudeYesNoEngine({"api_key": "sk-ant-..."})

print(engine.yes_or_no("Do you want me to set a timer?", "sure, go ahead"))  # True
print(engine.yes_or_no("Shall I call John?", "no, not now"))                 # False
print(engine.yes_or_no("Ready?", "what do you mean?"))                       # None
```

---

## ovos-memory-claude-plugin

**Class:** `ovos_claude.memory.ClaudeContextManager`
**OPM group:** `opm.agents.memory`
**Source:** `ovos_claude/memory.py:ClaudeContextManager`

Manages per-session conversation history in memory.
When the history exceeds `max_history` messages, the oldest half is summarised by Claude
and stored as a single SYSTEM message — `memory.py:ClaudeContextManager._compress_history`.

### Usage

```python
from ovos_claude.memory import ClaudeContextManager
from ovos_claude.chat import ClaudeChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

mem = ClaudeContextManager({
    "api_key": "sk-ant-...",
    "system_prompt": "You are a helpful assistant.",
    "max_history": 20,
})
engine = ClaudeChatEngine({"api_key": "sk-ant-..."})

session = "user-abc"

# Turn 1
ctx = mem.build_conversation_context("What is the capital of France?", session)
reply = engine.continue_chat(ctx)
mem.update_history([AgentMessage(MessageRole.USER, "What is the capital of France?"), reply], session)

# Turn 2 — history from turn 1 is included automatically
ctx = mem.build_conversation_context("And what language do they speak?", session)
reply = engine.continue_chat(ctx)
```

---

## ovos-utterance-transformer-claude-plugin

**Class:** `ovos_claude.transformers.ClaudeUtteranceTransformer`
**OPM group:** `opm.transformer.utterance`
**Source:** `ovos_claude/transformers.py:ClaudeUtteranceTransformer`

Runs after ASR, before intent matching (`ovos-core/ovos_core/transformers.py`).
Normalises informal or noisy speech to standard form.
Falls back to the original utterance on API error.

### Usage

```python
from ovos_claude.transformers import ClaudeUtteranceTransformer

t = ClaudeUtteranceTransformer(config={"api_key": "sk-ant-..."})
result, ctx = t.transform(["whats 2 plus 2 ya know"])
print(result)  # ["What is 2 plus 2?"]
```

---

## ovos-dialog-transformer-claude-plugin

**Class:** `ovos_claude.transformers.ClaudeDialogTransformer`
**OPM group:** `opm.transformer.dialog`
**Source:** `ovos_claude/transformers.py:ClaudeDialogTransformer`

Runs after skill response generation, before TTS synthesis.
Only invokes Claude when a `rewrite_prompt` is set (via config or `context["prompt"]`).
Falls back to the original dialog on API error.

### Usage

```python
from ovos_claude.transformers import ClaudeDialogTransformer

t = ClaudeDialogTransformer(config={
    "api_key": "sk-ant-...",
    "rewrite_prompt": "Rewrite in a cheerful, enthusiastic tone.",
})

result, ctx = t.transform("The forecast shows rain tomorrow.")
print(result)  # "Oh wow, rain is coming tomorrow — how exciting for the plants!"
```
