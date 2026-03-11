
# ovos-claude-plugin — Maintenance Report

## 2026-03-11 — Address PR #1 CodeRabbit and Bot Review Feedback

### Transparency Report
- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**:
  - `coverage.yml` — fixed branch trigger (`dev` → `dev, master, main`) and `coverage_source` (`ovos_claude_plugin` → `ovos_claude`)
  - `license_check.yml` — fixed branch trigger to also run on `master`/`main` PRs
  - `pip_audit.yml` — fixed branch trigger to also run on `master`/`main` PRs
  - `release-preview.yml` — corrected `package_name` from `ovos_claude_plugin` to `ovos-claude-plugin` (canonical PyPI name)
  - `ovos_claude/api.py` — changed `--system-prompt` to `--append-system-prompt` in `_build_cmd` and `stream_tokens` to preserve Claude Code's built-in instructions; added deadline enforcement across `stream_tokens` read loop to prevent indefinite blocking
  - `ovos_claude_persona/__init__.py` — fixed module docstring: config override key is `ovos-chat-claude-plugin` (was already correct in code but docs were ambiguous)
  - `FAQ.md` — updated ClaudeCodeChatEngine description to reflect `--append-system-prompt` and deadline enforcement
- **Oversight**: User-directed PR review; human reviews before push

## 2026-03-10 — Pre-Release Publishing Preparation

### Transparency Report
- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**:
  - Fixed `ovos_claude/version.py` — converted bare `START_VERSION_BLOCK`/`END_VERSION_BLOCK` tokens to Python comments (was causing `NameError` on import, blocking PyPI build)
  - Added `test/unittests/test_multimodal.py` — 3 tests for `ClaudeMultimodalChatEngine` (was 0% coverage)
  - Updated `AUDIT.md` — replaced stale 2026-03-09 entry with accurate pre-release findings
  - Updated `SUGGESTIONS.md` — replaced stale suggestions with current enhancement proposals
  - Updated `MAINTENANCE_REPORT.md` (this file)
- **Oversight**: User-directed publish preparation; human reviews before push

## 2026-03-09 — Agent Plugins Audit

### Transparency Report
- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**: Audited plugin, fixed CI workflows, added missing docs and LICENSE
- **Oversight**: User-directed audit of all agent plugins
