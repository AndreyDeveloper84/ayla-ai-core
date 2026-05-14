"""AnthropicCompletionAdapter — normalises Anthropic response into OpenAI shape.

Anthropic's Messages API returns::

    response.content = [
        {"type": "text", "text": "..."},
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
        ...
    ]
    response.usage = {"input_tokens": N, "output_tokens": M}

OpenAI's chat-completions returns::

    completion.choices[0].message.content = "..."
    completion.choices[0].message.tool_calls = [
        SimpleNamespace(id="...", function=SimpleNamespace(name="...", arguments="...")),
        ...
    ]
    completion.usage = SimpleNamespace(prompt_tokens=N, completion_tokens=M)

This adapter converts the former to a duck-typed equivalent of the latter so
``_parse_completion`` in the orchestrator stays provider-agnostic without an
explicit branch. Parallel tool calls (v0.7.2 / DRF-678) work transparently —
Anthropic returns multiple ``tool_use`` content blocks in a single message,
which we collect into the ``tool_calls`` list. Text+tool combos (Anthropic
emits both text and tool_use in the same response) merge content blocks
with ``\\n\\n`` between them before assignment, mirroring the rendering of an
OpenAI text response.

Argument shape difference: OpenAI sends tool arguments as a JSON-encoded
string; Anthropic sends them as a parsed dict. We re-serialize to keep the
contract uniform — ``tool_handlers.dispatch_tool_call`` does
``json.loads(arguments)`` and would fail on a raw dict.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


class AnthropicCompletionAdapter:
    """Normalise an Anthropic ``Message`` into OpenAI's chat-completions shape.

    Pass to :class:`AIConcierge(..., completion_adapter=AnthropicCompletionAdapter())`
    when the ``openai_client`` slot is wired to an Anthropic-compatible
    async client. The orchestrator hot path is unchanged.
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def normalize(self, completion: Any) -> Any:
        content_blocks = list(getattr(completion, "content", []) or [])

        # Anthropic represents text as `content[i].text` (type="text") and
        # tool calls as `content[i].input` / `content[i].name` (type="tool_use").
        # We merge text blocks into a single string (OpenAI puts the whole
        # assistant message in `.content`) and convert tool_use blocks into
        # SimpleNamespace mimicking OpenAI's ChatCompletionMessageToolCall.
        text_chunks: list[str] = []
        tool_calls: list[SimpleNamespace] = []
        for block in content_blocks:
            block_type = self._get(block, "type", "")
            if block_type == "text":
                text = self._get(block, "text", "") or ""
                if text:
                    text_chunks.append(text)
            elif block_type == "tool_use":
                tc_id = self._get(block, "id", "") or ""
                tc_name = self._get(block, "name", "") or ""
                tc_input = self._get(block, "input", {}) or {}
                # OpenAI expects `arguments` as a JSON string. Re-serialise
                # for the contract `tool_handlers.dispatch_tool_call` relies on.
                arguments = json.dumps(tc_input, ensure_ascii=False)
                tool_calls.append(
                    SimpleNamespace(
                        id=tc_id,
                        function=SimpleNamespace(name=tc_name, arguments=arguments),
                    )
                )

        merged_content = "\n\n".join(text_chunks) if text_chunks else ""

        # Anthropic usage uses input_tokens/output_tokens; map to OpenAI names.
        usage_obj = getattr(completion, "usage", None)
        if usage_obj is not None:
            prompt_tokens = self._get(usage_obj, "input_tokens", 0) or 0
            completion_tokens = self._get(usage_obj, "output_tokens", 0) or 0
            usage_ns: Any = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        else:
            usage_ns = None

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=merged_content or None,
                        tool_calls=tool_calls or None,
                    ),
                ),
            ],
            usage=usage_ns,
        )

    @staticmethod
    def _get(obj: Any, name: str, default: Any) -> Any:
        """Read ``name`` from ``obj`` whether it's a dict-like or attribute-like."""
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
