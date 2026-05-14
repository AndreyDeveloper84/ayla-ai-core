"""Tests for v0.8.0 / Arch-2 (DRF-686): PromptComposer + byte-identical guarantee."""
from __future__ import annotations

from datetime import date

import pytest

from ayla_ai_core.composer import PromptComposer
from ayla_ai_core.context import (
    SpecialistCandidate,
    build_specialist_context_from_candidates,
)
from ayla_ai_core.prompts import (
    FORMULA_TELA_VOICE,
    BrandVoiceConfig,
    Example,
    render_system_prompt,
)


@pytest.fixture
def master_context():
    """Realistic booking context used by FORMULA_TELA in prod."""
    candidates = [
        SpecialistCandidate(
            id=1, name="Анна", specialization="массаж",
            services=[(10, "массаж спины")],
        ),
        SpecialistCandidate(
            id=2, name="Борис", specialization="СПА",
            services=[(11, "СПА процедура")],
        ),
    ]
    return build_specialist_context_from_candidates(candidates, tenant_id="formula-tela")


class TestPromptComposerByteIdenticalToLegacy:
    """**Critical Arch-2 contract.** PromptComposer.render() with no section
    overrides MUST produce byte-identical output to render_system_prompt(...)
    for the booking-domain voices. This is enforced by 20+ existing replay
    fixtures in ai-bot-platform; divergence breaks production replay.
    """

    def test_no_overrides_byte_identical_to_legacy(self, master_context) -> None:
        kwargs = {
            "today": date(2026, 5, 14),
            "client_name": "Анна Иванова",
            "bookings_count": 3,
            "specialist_context": master_context,
            "extra_hint": "",
            "escape_for_format": True,
        }

        legacy_output = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **kwargs)
        composer_output = PromptComposer(voice=FORMULA_TELA_VOICE).render(**kwargs)

        assert composer_output == legacy_output, (
            "Arch-2 byte-identical guarantee broken — composer output diverged "
            "from legacy render_system_prompt. Existing replay fixtures will fail."
        )

    def test_with_brace_injection_in_client_name_byte_identical(
        self, master_context,
    ) -> None:
        """B4 anti-injection (v0.7.0) must survive the composer wrapping."""
        kwargs = {
            "today": date(2026, 5, 14),
            "client_name": "{evil}",
            "bookings_count": 0,
            "specialist_context": master_context,
            "extra_hint": "",
            "escape_for_format": True,
        }

        legacy = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **kwargs)
        composer = PromptComposer(voice=FORMULA_TELA_VOICE).render(**kwargs)

        assert legacy == composer
        # Sanity: the {evil} survives as literal (not template-substituted).
        assert "{evil}" in legacy


class TestPromptComposerSectionOverride:
    """Section overrides let non-booking domains rebuild the prompt without
    forking render_system_prompt."""

    def test_masters_summary_override_appears_in_output(
        self, master_context,
    ) -> None:
        custom_summary = "[CUSTOM FAQ CONTEXT — knowledge base chunks here]"
        composer = (
            PromptComposer(voice=FORMULA_TELA_VOICE)
            .with_section("masters_summary", custom_summary)
        )

        out = composer.render(
            today=date(2026, 5, 14),
            client_name="x",
            bookings_count=0,
            specialist_context=master_context,
        )

        # The override replaces the default `summary_text` in the rendered
        # prompt. Original `master_id=1 Анна` rendering should be gone.
        assert custom_summary in out
        assert "master_id=1 Анна" not in out

    def test_no_override_uses_context_summary(self, master_context) -> None:
        """Sanity: without override, legacy summary_text from context renders."""
        out = PromptComposer(voice=FORMULA_TELA_VOICE).render(
            today=date(2026, 5, 14),
            client_name="x",
            bookings_count=0,
            specialist_context=master_context,
        )
        # Default rendering includes the master_id=N format.
        assert "master_id=1" in out
        assert "Анна" in out

    def test_section_override_does_not_mutate_caller_context(
        self, master_context,
    ) -> None:
        """SpecialistContext is frozen; composer must not mutate it via the
        override mechanism. Confirms the dataclasses.replace contract."""
        original_summary = master_context.summary_text

        PromptComposer(voice=FORMULA_TELA_VOICE).with_section(
            "masters_summary", "MUTATED",
        ).render(
            today=date(2026, 5, 14),
            client_name="x",
            bookings_count=0,
            specialist_context=master_context,
        )

        # The caller's original context must be untouched.
        assert master_context.summary_text == original_summary


class TestPromptComposerExamples:
    """Composer.with_examples merges into voice_config.examples for the
    duration of a single render."""

    def test_composer_examples_appended_to_voice_examples(
        self, master_context,
    ) -> None:
        custom = [Example(user="Что такое СПА?", assistant="Это процедура...")]
        voice_with_no_examples = BrandVoiceConfig(
            assistant_name="Тест",
            business_name="Тест Бизнес",
            business_address="Тест адрес",
            domain="test",
            off_topic_redirect="—",
        )
        composer = PromptComposer(voice=voice_with_no_examples).with_examples(custom)

        out = composer.render(
            today=date(2026, 5, 14),
            client_name="x",
            bookings_count=0,
            specialist_context=master_context,
        )

        # Custom example must appear in rendered output.
        assert "Что такое СПА?" in out

    def test_fluent_chaining_returns_self(self, master_context) -> None:
        composer = PromptComposer(voice=FORMULA_TELA_VOICE)
        same = composer.with_examples([]).with_section("x", "y")
        assert same is composer


class TestPromptComposerInitState:
    """Constructor + read-only property contracts."""

    def test_default_examples_is_empty(self) -> None:
        composer = PromptComposer(voice=FORMULA_TELA_VOICE)
        assert composer.examples == ()

    def test_examples_property_is_tuple_immutable(self) -> None:
        """The `examples` property returns a tuple — caller mutation of the
        returned object MUST NOT affect composer state."""
        composer = PromptComposer(
            voice=FORMULA_TELA_VOICE,
            examples=[Example(user="a", assistant="b")],
        )
        snapshot = composer.examples
        assert isinstance(snapshot, tuple)
        # Can't mutate via subscript — tuples are immutable.
        with pytest.raises((TypeError, AttributeError)):
            snapshot[0] = Example(user="c", assistant="d")  # type: ignore[index]

    def test_sections_property_returns_snapshot_copy(self) -> None:
        """Mutating the returned dict must not change composer state."""
        composer = PromptComposer(voice=FORMULA_TELA_VOICE).with_section("a", "b")
        sections = composer.sections
        sections["a"] = "MUTATED"
        # Internal state unchanged.
        assert composer.sections == {"a": "b"}
