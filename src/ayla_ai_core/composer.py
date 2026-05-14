"""PromptComposer — pluggable section-based system prompt builder (v0.8.0).

Replaces the implicit-template pattern of :func:`render_system_prompt` for
consumers who need to render prompts for **non-booking** domains (FAQ
chunks, support tickets, knowledge-base summaries — anything where the
hardcoded ``masters_summary`` slot doesn't fit). The composer is
fluent / chainable; sections can be added or replaced before
:meth:`render` runs.

**Byte-identical guarantee:** for the FORMULA_TELA_VOICE +
AYLA_MARKETPLACE_VOICE paths used by ``ai-bot-platform`` and the legacy
``mysite/maxbot``, calling :meth:`PromptComposer.render` with the same
inputs produces byte-identical output to ``render_system_prompt(...)``.
A golden-file test in ``tests/test_composer.py`` pins this — Arch-2
must NOT diverge from the legacy booking-domain output without a
deliberate v0.9.0 minor bump.

**Future flexibility:** ``with_section("custom_block", "...")`` injects
arbitrary content. The composer's pluggable design lets FAQ-style
consumers replace the entire prompt body without touching
``render_system_prompt`` itself. v0.9.0 will move ``render_system_prompt``
into a thin wrapper that calls ``PromptComposer.default_for_voice(...)``;
v0.8.0 keeps both side-by-side for migration.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from ayla_ai_core.context import SpecialistContext
from ayla_ai_core.prompts import BrandVoiceConfig, Example, render_system_prompt


class PromptComposer:
    """Fluent builder for system prompts. v0.8.0 / Arch-2 (DRF-686).

    Default rendering delegates to :func:`render_system_prompt` for the
    booking domain (FORMULA_TELA / AYLA_MARKETPLACE). Sections override
    or augment the legacy template's slots; passing only ``voice`` +
    ``specialist_context`` reproduces v0.7.x output byte-for-byte.

    Usage::

        composer = (
            PromptComposer(voice=FORMULA_TELA_VOICE)
            .with_examples([Example(user=..., assistant=...)])
            .with_section("masters_summary", ctx.summary_text)
        )
        system_prompt = composer.render(
            today=date.today(),
            client_name="Анна",
            bookings_count=3,
            specialist_context=ctx,
        )
    """

    def __init__(
        self,
        *,
        voice: BrandVoiceConfig,
        examples: list[Example] | None = None,
    ) -> None:
        self._voice = voice
        self._examples: list[Example] = list(examples) if examples else []
        # Section override map. Keys not present here fall back to the
        # default rendering inside `render_system_prompt`. Values are raw
        # strings (already-escaped if the caller pre-escaped, since the
        # default `render_system_prompt(escape_for_format=True)` will
        # double-escape on its own pass — see Arch-5's
        # OpenAIPassthroughAdapter parallel — caller controls the kwarg).
        self._section_overrides: dict[str, str] = {}

    @property
    def voice(self) -> BrandVoiceConfig:
        return self._voice

    @property
    def examples(self) -> tuple[Example, ...]:
        return tuple(self._examples)

    @property
    def sections(self) -> dict[str, str]:
        """Snapshot of current section overrides. Read-only — use
        :meth:`with_section` to add."""
        return dict(self._section_overrides)

    def with_examples(self, examples: list[Example]) -> PromptComposer:
        """Replace the examples list. Returns ``self`` for chaining."""
        self._examples = list(examples)
        return self

    def with_section(self, name: str, content: str) -> PromptComposer:
        """Set an override for a named section (e.g. ``"masters_summary"``,
        or a custom block your downstream template handles). Returns
        ``self`` for chaining.

        v0.8.0 recognised sections: ``masters_summary`` (overrides
        ``specialist_context.summary_text`` in the rendered prompt). Other
        names are accepted and stored but currently ignored by the booking
        template; v0.9.0 will expose them via a section-aware template
        engine.
        """
        self._section_overrides[name] = content
        return self

    def render(
        self,
        *,
        today: date,
        client_name: str,
        bookings_count: int,
        specialist_context: SpecialistContext[Any],
        extra_hint: str = "",
        escape_for_format: bool = True,
    ) -> str:
        """Render the system prompt. v0.8.0 / Arch-2: delegates to the
        legacy :func:`render_system_prompt` for FORMULA_TELA / AYLA paths
        with optional ``masters_summary`` section override applied.

        Byte-identical to ``render_system_prompt(...)`` when no section
        overrides are set and ``examples=self._examples`` matches.
        """
        # If the caller supplied a `masters_summary` override, swap it into
        # the SpecialistContext's `summary_text` slot for this render.
        # SpecialistContext is frozen; produce a shallow copy with the
        # replacement so the original passed-in context is not mutated
        # (preserves replay-determinism + thread safety).
        if "masters_summary" in self._section_overrides:
            from dataclasses import replace

            ctx_for_render = replace(
                specialist_context,
                summary_text=self._section_overrides["masters_summary"],
            )
        else:
            ctx_for_render = specialist_context

        # Append composer-level examples into the voice's example list for
        # this single render call. We rebuild the voice config with the
        # combined list — `voice_config.examples` flows into the template
        # via `_render_examples_block`.
        if self._examples:
            from dataclasses import replace

            combined_examples = [*self._voice.examples, *self._examples]
            voice_for_render = replace(self._voice, examples=combined_examples)
        else:
            voice_for_render = self._voice

        return render_system_prompt(
            today=today,
            client_name=client_name,
            bookings_count=bookings_count,
            specialist_context=ctx_for_render,
            voice_config=voice_for_render,
            extra_hint=extra_hint,
            escape_for_format=escape_for_format,
        )
