"""
ovos-claude-persona — ships the default Claude persona config dict.

The persona dict is loaded by PersonaService via the ``opm.plugin.persona``
entry point.  Users can activate Claude by name ("Hey, ask Claude...") or
set it as their default persona.

To override settings, add an ``ovos-chat-claude-plugin`` key nested under
``persona`` → ``Claude`` in ``~/.config/mycroft/mycroft.conf``.
"""
CLAUDE_PERSONA = {
    "name": "Claude",
    # The chat engine plugin to use
    "solvers": ["ovos-chat-claude-plugin"],
    # Brief, voice-friendly system prompt
    "system_prompt": (
        "You are Claude, a helpful voice assistant made by Anthropic. "
        "Give concise, clear answers suitable for text-to-speech. "
        "Avoid markdown, bullet points, and special characters."
    ),
    # Plugin-specific config (can be overridden in mycroft.conf)
    "ovos-chat-claude-plugin": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 200,
        "temperature": 0.7,
    },
}
