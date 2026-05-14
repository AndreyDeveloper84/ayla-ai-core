"""CompletionAdapter Protocol — pluggable response-shape transformer (v0.8.0).

The orchestrator parses ``completion.choices[0].message`` and walks
``message.tool_calls`` to extract tool-call payloads. That assumption is
hardcoded to OpenAI's chat-completions response shape, which leaks the
provider into the orchestrator. v0.8.0 introduces this Protocol so each
provider's adapter normalises its native response into a duck-typed
"OpenAI-shaped" object that the downstream parser can read without change.

Backward compat: when no adapter is supplied, the orchestrator behaves as
v0.7.x did (passthrough, expects OpenAI shape).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CompletionAdapter(Protocol):
    """Transform a provider's native completion into the OpenAI-shaped
    duck-typed object the orchestrator expects.

    The returned object MUST expose:

    - ``.choices[0].message.content`` — ``str | None``
    - ``.choices[0].message.tool_calls`` — ``list`` or ``None``; each entry has
      ``.id`` (str), ``.function.name`` (str), ``.function.arguments`` (JSON str)
    - ``.usage.prompt_tokens`` / ``.usage.completion_tokens`` — ``int``
      (or ``None`` if the provider doesn't expose them)

    The simplest implementation (when consuming OpenAI directly) is the
    identity transform: see :class:`OpenAIPassthroughAdapter`. Anthropic
    consumers use :class:`~ayla_ai_core.providers.anthropic.AnthropicCompletionAdapter`.

    Implementations must NOT do network I/O — the adapter runs synchronously
    inside the orchestrator hot path after ``await openai_client.chat.completions.create``.
    """

    def normalize(self, completion: Any) -> Any:
        """Return an OpenAI-shaped duck-typed object. ``completion`` is the
        raw provider response as returned by the async client supplied to
        :class:`AIConcierge`."""
        ...

    @property
    def provider_name(self) -> str:
        """Identifier surfaced in :class:`ChatResponseDTO.provider`.

        Examples: ``"openai"`` (default), ``"anthropic"``, ``"azure-openai"``.
        Consumers use this for per-provider dashboards / cost attribution.
        """
        ...


class OpenAIPassthroughAdapter:
    """No-op adapter — OpenAI-shaped completions pass through unchanged.

    Used as the default when :class:`AIConcierge` is constructed without an
    explicit ``completion_adapter``. Preserves v0.7.x behaviour byte-for-byte.
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def normalize(self, completion: Any) -> Any:
        return completion
