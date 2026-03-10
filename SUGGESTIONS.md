Last Edit: Claude Sonnet 4.6 - 2026-03-10 - Motive: Replace stale suggestions with current proposals

# ovos-claude-plugin — Suggestions

## 1. Bump minimum Python to 3.10
- **Problem**: `requires-python = ">=3.9"` but OVOS workspace standard (AGENTS.md §2) targets 3.10+; Python 3.9 reached EOL Oct 2025
- **Proposed solution**: Change `pyproject.toml` to `requires-python = ">=3.10"` and remove 3.9 from CI matrix
- **Impact**: Low — no 3.9-incompatible syntax is used; simplifies maintenance

## 2. Increase API error coverage in test_api.py
- **Problem**: `ovos_claude/api.py` has 80% coverage; lines 78–88 (`_get_client` exception path) and 244–246 (streaming error handler) are not exercised
- **Proposed solution**: Add parametrised tests for `ImportError` (missing `anthropic`) and streaming `APIError`
- **Impact**: Prevents silent regressions on API key / SDK failures

## 3. Add streaming support to ClaudeChatEngine
- **Problem**: `ClaudeChatEngine.continue_chat()` blocks until the full response is received; long answers feel slow in voice context
- **Proposed solution**: Expose an optional `stream=True` path that yields sentence fragments via `sentence_stream.SentenceBoundaryDetector`; `SentenceBoundaryDetector` is already a dependency
- **Impact**: Medium — improves first-response latency for voice; requires changes to `ovos_claude/chat.py` and `ovos_claude/api.py`

## 4. Tool-use / function-calling engine
- **Problem**: No OVOS plugin exposes Claude's tool-use API
- **Proposed solution**: Add `ClaudeToolEngine` entry point under a new `opm.agents.tool` group (pending OPM support); wrap `client.messages.create(tools=[...])` with the same lazy-init pattern as `AnthropicClient`
- **Impact**: High — unlocks agentic skills; depends on upstream OPM adding `opm.agents.tool` entry-point group
