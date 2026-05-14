"""Provider adapters — pluggable response-shape transformations (v0.8.0+).

ayla v0.7.x hard-expected OpenAI completion shape
(``completion.choices[0].message.tool_calls[i].function.{name,arguments}``).
Anthropic emits a different shape — ``response.content[i].type=tool_use``
with ``content[i].{name,input}`` for tool calls and ``content[i].text`` for
text. v0.8.0 introduces a :class:`CompletionAdapter` Protocol so the
orchestrator can stay shape-agnostic; providers convert their native
response into a `_NormalizedCompletion` that mirrors the OpenAI shape (the
de-facto wire format ``ayla.tool_handlers`` parses).

Default behaviour is unchanged for OpenAI consumers — when no
``completion_adapter`` is passed to :class:`AIConcierge`, the response is
parsed as OpenAI-shaped (zero migration for v0.7.x consumers). Anthropic
consumers pass :class:`AnthropicCompletionAdapter` and write their
``openai_client`` slot to an Anthropic-compatible async client (or a
provider wrapper that conforms to the same ``chat.completions.create``
protocol).
"""
from __future__ import annotations

from ayla_ai_core.providers.anthropic import AnthropicCompletionAdapter
from ayla_ai_core.providers.base import CompletionAdapter, OpenAIPassthroughAdapter

__all__ = [
    "AnthropicCompletionAdapter",
    "CompletionAdapter",
    "OpenAIPassthroughAdapter",
]
