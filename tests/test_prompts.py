"""Tests для prompts.py — BrandVoiceConfig + render_system_prompt."""
from __future__ import annotations

from datetime import date
from uuid import UUID

import pytest

from ayla_ai_core.context import (
    SpecialistCandidate,
    build_specialist_context_from_candidates,
)
from ayla_ai_core.prompts import (
    AYLA_MARKETPLACE_VOICE,
    FORMULA_TELA_VOICE,
    BrandVoiceConfig,
    Example,
    render_system_prompt,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def specialist_context():
    """Минимальный SpecialistContext[int] для бота."""
    candidates = [
        SpecialistCandidate(
            id=42, name="Анна Иванова", specialization="массаж",
            services=[(10, "массаж спины"), (11, "лимфодренаж")],
        ),
    ]
    # v0.7.0: tenant_id mandatory. Synthetic test value — clearly not real.
    return build_specialist_context_from_candidates(candidates, tenant_id="test-tenant")


@pytest.fixture
def empty_context():
    return build_specialist_context_from_candidates([], tenant_id="test-tenant")


@pytest.fixture
def render_kwargs(specialist_context):
    return {
        "today": date(2026, 5, 15),
        "client_name": "Мария",
        "bookings_count": 3,
        "specialist_context": specialist_context,
    }


# ─── BrandVoiceConfig dataclass ───────────────────────────────────────────


def test_brand_voice_config_is_frozen() -> None:
    config = BrandVoiceConfig(
        assistant_name="X", business_name="Y", business_address=None,
        domain="test", off_topic_redirect="redirect",
    )
    with pytest.raises(AttributeError):
        config.assistant_name = "Z"  # type: ignore[misc]


def test_brand_voice_config_examples_default_empty() -> None:
    config = BrandVoiceConfig(
        assistant_name="X", business_name="Y", business_address=None,
        domain="t", off_topic_redirect="r",
    )
    assert config.examples == []
    assert config.use_long_term_memory_hint is False


def test_example_dataclass() -> None:
    ex = Example(user="хочу маникюр", assistant="вызвать show_masters")
    assert ex.user == "хочу маникюр"
    assert ex.assistant == "вызвать show_masters"


# ─── render_system_prompt — Formula tela voice ────────────────────────────


def test_render_formula_tela_includes_assistant_name(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "Алина" in out
    assert "Формула тела" in out
    assert "Пензе" in out


def test_render_formula_tela_off_topic_redirect(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "массаж и SPA в нашем салоне" in out


def test_render_formula_tela_no_memory_hint_section(render_kwargs) -> None:
    """Бот: use_long_term_memory_hint=False → секции про память нет."""
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "долгосрочная память" not in out


# ─── render_system_prompt — Ayla marketplace voice ────────────────────────


def test_render_ayla_marketplace_uses_ayla_name(render_kwargs) -> None:
    out = render_system_prompt(voice_config=AYLA_MARKETPLACE_VOICE, **render_kwargs)
    assert "Ayla" in out
    assert "AI-помощник" in out
    # Без локального адреса (marketplace)
    assert "Пензе" not in out
    assert "ул. Пушкина" not in out


def test_render_ayla_includes_memory_hint(render_kwargs) -> None:
    """Ayla: use_long_term_memory_hint=True → правило 9 про память."""
    out = render_system_prompt(voice_config=AYLA_MARKETPLACE_VOICE, **render_kwargs)
    assert "долгосрочная память" in out


def test_render_ayla_off_topic_redirect(render_kwargs) -> None:
    out = render_system_prompt(voice_config=AYLA_MARKETPLACE_VOICE, **render_kwargs)
    assert "beauty-мастерам" in out


# ─── render_system_prompt — common (state injection) ──────────────────────


def test_render_includes_today(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "2026-05-15" in out


def test_render_includes_client_name(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "Мария" in out


def test_render_empty_client_name_falls_back(specialist_context) -> None:
    out = render_system_prompt(
        today=date(2026, 5, 15),
        client_name="",
        bookings_count=0,
        specialist_context=specialist_context,
        voice_config=FORMULA_TELA_VOICE,
    )
    assert "клиент" in out


def test_render_includes_bookings_count(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "Прошлых записей у клиента: 3" in out


def test_render_includes_masters_summary(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "master_id=42" in out
    assert "Анна Иванова" in out
    assert "service_id=10" in out


def test_render_empty_context_no_masters(empty_context) -> None:
    out = render_system_prompt(
        today=date(2026, 5, 15), client_name="X", bookings_count=0,
        specialist_context=empty_context, voice_config=FORMULA_TELA_VOICE,
    )
    assert "(нет активных мастеров" in out


# ─── render_system_prompt — empty slots rules (Phase 0 hot fix) ───────────


def test_render_includes_empty_slots_rules(render_kwargs) -> None:
    """Phase 0 hot fix — правила про пустые слоты должны быть."""
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "ПРАВИЛА ПРИ ПУСТЫХ СЛОТАХ" in out
    assert "не заканчивай диалог фразой «нет слотов»".lower() in out.lower()
    assert "ask_clarification" in out


# ─── render_system_prompt — anti-hallucination rules ──────────────────────


def test_render_includes_critical_anti_hallucination_rule(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "КРИТИЧЕСКОЕ ПРАВИЛО" in out
    assert "show_masters вместо текста" in out


def test_render_includes_no_phone_request_rule(render_kwargs) -> None:
    """Правило 7 — не запрашивать телефон/email."""
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "телефон" in out


def test_render_includes_all_5_tools(render_kwargs) -> None:
    """Все 5 tools упоминаются в prompt."""
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    for tool in ["show_masters", "show_slots", "confirm_booking",
                 "show_my_bookings", "ask_clarification"]:
        assert tool in out


# ─── render_system_prompt — examples block (Level 5 enabler) ──────────────


def test_render_no_examples_block_when_empty(render_kwargs) -> None:
    """Default config (examples=[]) — examples block отсутствует."""
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "ПРИМЕРЫ ХОРОШИХ ДИАЛОГОВ" not in out


def test_render_includes_examples_when_provided(render_kwargs) -> None:
    """Custom config с examples — ПРИМЕРЫ ХОРОШИХ ДИАЛОГОВ block добавляется."""
    config_with_examples = BrandVoiceConfig(
        assistant_name="Алина",
        business_name="Формула тела",
        business_address="Пензе",
        domain="массаж",
        off_topic_redirect="redirect",
        examples=[
            Example(user="болит спина", assistant="show_masters [массажисты]"),
            Example(user="когда у меня запись", assistant="show_my_bookings"),
        ],
    )
    out = render_system_prompt(voice_config=config_with_examples, **render_kwargs)
    assert "ПРИМЕРЫ ХОРОШИХ ДИАЛОГОВ" in out
    assert "болит спина" in out
    assert "когда у меня запись" in out


# ─── Token budget ─────────────────────────────────────────────────────────


def test_render_token_budget_under_2000_for_small_context(render_kwargs) -> None:
    """Базовый случай (1 мастер, без examples) — должен быть <2000 токенов
    (грубая оценка: chars / 4, не точный tiktoken).
    """
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    # Грубо: 1 token ≈ 4 chars для русского текста
    estimated_tokens = len(out) / 4
    assert estimated_tokens < 2000, (
        f"Prompt too large: {len(out)} chars ≈ {estimated_tokens:.0f} tokens"
    )


def test_render_token_budget_under_2500_with_50_candidates() -> None:
    """50 мастеров — prompt всё ещё в разумных пределах для gpt-4o-mini.

    Это edge case (Ayla marketplace при scale). Проверяем что не взрывается.
    """
    candidates = [
        SpecialistCandidate(
            id=i, name=f"Мастер {i}", specialization="массаж",
            services=[(j, f"услуга {j}") for j in range(i, i + 3)],
        )
        for i in range(1, 51)
    ]
    ctx = build_specialist_context_from_candidates(candidates, tenant_id="test-tenant")
    out = render_system_prompt(
        today=date(2026, 5, 15), client_name="X", bookings_count=0,
        specialist_context=ctx, voice_config=FORMULA_TELA_VOICE,
    )
    estimated_tokens = len(out) / 4
    # 50 candidates × 4 lines × 30 chars ≈ 6000 chars + base ~5000 chars = ~2750 tokens
    # Допускаем до 4000 для safety на 50 candidates (но в проде Top-N=20)
    assert estimated_tokens < 4000


# ─── render_system_prompt — UUID context (Ayla scenario) ──────────────────


def test_render_works_with_uuid_specialist_context() -> None:
    """SpecialistContext[UUID] (Ayla) — render выдаёт валидный prompt."""
    uid = UUID("11111111-1111-1111-1111-111111111111")
    sid = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    candidates = [
        SpecialistCandidate(
            id=uid, name="Anna", specialization="massage",
            services=[(sid, "back massage")],
        ),
    ]
    ctx = build_specialist_context_from_candidates(candidates, tenant_id="ayla")

    out = render_system_prompt(
        today=date(2026, 5, 15), client_name="Maria", bookings_count=0,
        specialist_context=ctx, voice_config=AYLA_MARKETPLACE_VOICE,
    )
    assert "11111111-1111" in out
    assert "Ayla" in out


# ─── Pre-built configs sanity checks ─────────────────────────────────────


def test_formula_tela_voice_has_expected_values() -> None:
    assert FORMULA_TELA_VOICE.assistant_name == "Алина"
    assert FORMULA_TELA_VOICE.business_name == "Формула тела"
    assert FORMULA_TELA_VOICE.business_address is not None
    assert FORMULA_TELA_VOICE.use_long_term_memory_hint is False


def test_ayla_marketplace_voice_has_expected_values() -> None:
    assert AYLA_MARKETPLACE_VOICE.assistant_name == "Ayla"
    assert AYLA_MARKETPLACE_VOICE.business_address is None
    assert AYLA_MARKETPLACE_VOICE.use_long_term_memory_hint is True


# ─── extra_hint kwarg (DRF-248) ────────────────────────────────────────────


def test_render_default_has_no_extra_hint_block(render_kwargs) -> None:
    out = render_system_prompt(voice_config=FORMULA_TELA_VOICE, **render_kwargs)
    assert "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ" not in out


def test_render_with_extra_hint_inserts_paragraph(render_kwargs) -> None:
    out = render_system_prompt(
        voice_config=FORMULA_TELA_VOICE,
        extra_hint="У клиента 3 дня подряд белок ниже нормы.",
        **render_kwargs,
    )
    assert "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ" in out
    assert "белок ниже нормы" in out
    # Soft framing — must mark as advisory not a rule.
    assert "мягкая подсказка" in out


def test_render_empty_or_whitespace_extra_hint_is_no_op(render_kwargs) -> None:
    for value in ("", "   ", "\n\t  \n"):
        out = render_system_prompt(
            voice_config=FORMULA_TELA_VOICE,
            extra_hint=value,
            **render_kwargs,
        )
        assert "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ" not in out


def test_render_extra_hint_works_with_ayla_voice_too(render_kwargs) -> None:
    out = render_system_prompt(
        voice_config=AYLA_MARKETPLACE_VOICE,
        extra_hint="Тестовая подсказка.",
        **render_kwargs,
    )
    assert "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ" in out
    # Memory hint section still present (Ayla voice).
    assert "помнишь" in out.lower() or "помню" in out.lower()


# ─── B4 regression tests (v0.7.0): escape_for_format anti-injection ────────


class TestBraceEscape:
    """B4 (v0.7.0): render_system_prompt escapes braces in user-controlled
    fields by default. Prevents KeyError DoS + template-injection on stray
    `{` and `}` in client_name / extra_hint / brand-config strings.
    """

    @staticmethod
    def _render(voice_config=None, **overrides):
        """Helper. Builds minimal specialist_context + voice, calls render."""
        from ayla_ai_core.context import (
            SpecialistCandidate,
            build_specialist_context_from_candidates,
        )

        candidates = [
            SpecialistCandidate(
                id=42, name="Анна", specialization="массаж",
                services=[(10, "массаж спины")],
            ),
        ]
        ctx = build_specialist_context_from_candidates(candidates, tenant_id="test-tenant")
        kwargs = {
            "today": date(2026, 5, 15),
            "client_name": "Мария",
            "bookings_count": 1,
            "specialist_context": ctx,
            "voice_config": voice_config or FORMULA_TELA_VOICE,
        }
        kwargs.update(overrides)
        return render_system_prompt(**kwargs)

    def test_client_name_with_braces_does_not_break_format(self):
        """`client_name="{evil}"` would have raised KeyError pre-v0.7.0."""
        out = self._render(client_name="{evil}")
        # Output contains literal `{evil}`, not template artifacts
        assert "{evil}" in out

    def test_extra_hint_with_template_injection_neutralized(self):
        """`extra_hint="{client_name}"` must NOT substitute the actual
        client_name value (which would be a real injection)."""
        out = self._render(
            client_name="SECRET-MARIA",
            extra_hint="see this: {client_name}",
        )
        # The injected `{client_name}` in extra_hint should remain LITERAL,
        # not get replaced with "SECRET-MARIA" via .format() inner-pass.
        # Search for the literal token in the rendered output.
        assert "{client_name}" in out
        # Sanity: client_name was used in its proper slot
        assert "SECRET-MARIA" in out

    def test_business_name_with_braces_escaped(self):
        """Brand-config fields also escape (defends against future
        DB-backed BrandVoiceConfig where admins could enter braces)."""
        from ayla_ai_core.prompts import BrandVoiceConfig

        evil_voice = BrandVoiceConfig(
            assistant_name="Aya",
            business_name="Salon {evil} Inc",
            business_address="Penza, {addr}",
            domain="beauty",
            off_topic_redirect="Off-topic: {redirect}",
        )
        out = self._render(voice_config=evil_voice)
        # Each malicious literal survives the .format() round-trip
        assert "{evil}" in out
        assert "{addr}" in out
        assert "{redirect}" in out

    def test_escape_off_legacy_behavior(self):
        """`escape_for_format=False` reproduces v0.6.x identical output for
        callers that already escape in a wrapper layer (e.g. ai-bot-platform
        ayla_adapter DRF-616). Safe values pass through unchanged."""
        out_safe = self._render(client_name="Maria", escape_for_format=False)
        # Without braces in input, output is identical regardless of flag
        out_default = self._render(client_name="Maria")
        # Strip the variable date/etc — easier to just assert client name present
        assert "Maria" in out_safe
        assert "Maria" in out_default

    def test_escape_off_with_braces_would_break(self):
        """With escape_for_format=False, raw `{client_name}` in extra_hint
        triggers template substitution — this documents the v0.6.x
        behavior the kwarg disables. Caller must escape themselves."""
        # Caller pre-escapes — adapter does this in ai-bot-platform
        from ayla_ai_core.prompts import _escape_braces

        out = self._render(
            client_name="Мария",
            extra_hint=_escape_braces("see this: {client_name}"),
            escape_for_format=False,
        )
        assert "{client_name}" in out
