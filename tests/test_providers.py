"""Tests for v0.8.0 / Arch-5 (DRF-689): CompletionAdapter + Anthropic adapter."""
from __future__ import annotations

import json
from types import SimpleNamespace

from ayla_ai_core.providers import (
    AnthropicCompletionAdapter,
    OpenAIPassthroughAdapter,
)
from ayla_ai_core.providers.base import CompletionAdapter


class TestOpenAIPassthrough:
    def test_provider_name_is_openai(self) -> None:
        assert OpenAIPassthroughAdapter().provider_name == "openai"

    def test_normalize_returns_input_unchanged(self) -> None:
        adapter = OpenAIPassthroughAdapter()
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
        assert adapter.normalize(completion) is completion

    def test_is_completion_adapter(self) -> None:
        """OpenAIPassthroughAdapter must satisfy the Protocol at runtime."""
        assert isinstance(OpenAIPassthroughAdapter(), CompletionAdapter)


class TestAnthropicCompletionAdapter:
    """Normalising Anthropic's content-block response into OpenAI shape."""

    def test_provider_name_is_anthropic(self) -> None:
        assert AnthropicCompletionAdapter().provider_name == "anthropic"

    def test_text_only_response(self) -> None:
        """Single text block → message.content is the text, tool_calls None."""
        anthropic_resp = SimpleNamespace(
            content=[{"type": "text", "text": "Привет"}],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        out = AnthropicCompletionAdapter().normalize(anthropic_resp)

        assert out.choices[0].message.content == "Привет"
        assert out.choices[0].message.tool_calls is None
        assert out.usage.prompt_tokens == 10
        assert out.usage.completion_tokens == 5

    def test_single_tool_use_block(self) -> None:
        """Single tool_use → message.tool_calls[0].function.{name,arguments}."""
        anthropic_resp = SimpleNamespace(
            content=[
                {
                    "type": "tool_use",
                    "id": "tu_01",
                    "name": "show_masters",
                    "input": {"master_ids": [1, 2], "explanation": "ok"},
                },
            ],
            usage=SimpleNamespace(input_tokens=20, output_tokens=15),
        )
        out = AnthropicCompletionAdapter().normalize(anthropic_resp)

        tc = out.choices[0].message.tool_calls
        assert tc is not None and len(tc) == 1
        assert tc[0].id == "tu_01"
        assert tc[0].function.name == "show_masters"
        # Anthropic sends arguments as dict; adapter must re-serialise to JSON
        # string so tool_handlers.dispatch_tool_call (which does json.loads)
        # can parse without crashing on a raw dict.
        assert isinstance(tc[0].function.arguments, str)
        parsed = json.loads(tc[0].function.arguments)
        assert parsed == {"master_ids": [1, 2], "explanation": "ok"}

    def test_parallel_tool_use_blocks(self) -> None:
        """Multiple tool_use blocks → parallel tool_calls (v0.7.2 / DRF-678)."""
        anthropic_resp = SimpleNamespace(
            content=[
                {"type": "tool_use", "id": "a", "name": "show_masters", "input": {"x": 1}},
                {"type": "tool_use", "id": "b", "name": "show_slots", "input": {"y": 2}},
                {"type": "tool_use", "id": "c", "name": "show_my_bookings", "input": {"z": 3}},
            ],
            usage=None,
        )
        out = AnthropicCompletionAdapter().normalize(anthropic_resp)

        tc = out.choices[0].message.tool_calls
        assert tc is not None and len(tc) == 3
        assert [t.function.name for t in tc] == ["show_masters", "show_slots", "show_my_bookings"]
        assert [t.id for t in tc] == ["a", "b", "c"]

    def test_text_and_tool_use_combined(self) -> None:
        """Anthropic emits both text AND tool_use in a single response —
        merge text blocks with \\n\\n, collect tool_use blocks separately."""
        anthropic_resp = SimpleNamespace(
            content=[
                {"type": "text", "text": "Сначала покажу мастеров."},
                {"type": "tool_use", "id": "tu_1", "name": "show_masters", "input": {}},
                {"type": "text", "text": "Потом уточню детали."},
            ],
            usage=None,
        )
        out = AnthropicCompletionAdapter().normalize(anthropic_resp)

        assert out.choices[0].message.content == (
            "Сначала покажу мастеров.\n\nПотом уточню детали."
        )
        assert out.choices[0].message.tool_calls is not None
        assert out.choices[0].message.tool_calls[0].function.name == "show_masters"

    def test_no_usage_returns_none_usage(self) -> None:
        """When provider doesn't expose usage, telemetry stays None — DTO
        defaults handle the gap (tokens_in/out = 0)."""
        anthropic_resp = SimpleNamespace(
            content=[{"type": "text", "text": "ok"}],
            usage=None,
        )
        out = AnthropicCompletionAdapter().normalize(anthropic_resp)
        assert out.usage is None

    def test_empty_content_yields_none_content(self) -> None:
        """No text + no tool_use → both content and tool_calls are falsy."""
        anthropic_resp = SimpleNamespace(content=[], usage=None)
        out = AnthropicCompletionAdapter().normalize(anthropic_resp)
        assert out.choices[0].message.content is None
        assert out.choices[0].message.tool_calls is None

    def test_dict_style_content_blocks_also_work(self) -> None:
        """Anthropic SDK exposes content blocks as objects with attributes
        in some versions and as dicts in others. Adapter must handle both."""
        anthropic_resp = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="object-style"),
                {"type": "text", "text": "dict-style"},
            ],
            usage=None,
        )
        out = AnthropicCompletionAdapter().normalize(anthropic_resp)
        assert "object-style" in out.choices[0].message.content
        assert "dict-style" in out.choices[0].message.content

    def test_is_completion_adapter(self) -> None:
        """Anthropic adapter must satisfy the runtime Protocol check."""
        assert isinstance(AnthropicCompletionAdapter(), CompletionAdapter)
