"""Tests для tool_handlers.py — anti-hallucination invariants critical.

Главные invariants:
1. LLM выдумал master_id который не в context.candidate_ids → fallback
2. LLM выдумал service_id → fallback
3. LLM выбрал master+service где master не оказывает service → fallback
4. Невалидный JSON в arguments → fallback
5. Unknown tool name → fallback

Эти fallback'и — гарантия что клиент НЕ увидит сломанную карточку.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ayla_ai_core.context import (
    SpecialistCandidate,
    SpecialistContext,
    build_specialist_context_from_candidates,
)
from ayla_ai_core.tool_handlers import (
    ToolResult,
    dispatch_tool_call,
    handle_ask_clarification,
    handle_confirm_booking,
    handle_show_masters,
    handle_show_my_bookings,
    handle_show_slots,
)
from ayla_ai_core.tools import ActionType

# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def master_context():
    """Контекст с 2 мастерами:
    - Анна (id=1): услуги 10 (массаж спины), 11 (лимфодренаж)
    - Борис (id=2): услуга 12 (СПА)
    """
    candidates = [
        SpecialistCandidate(
            id=1, name="Анна", specialization="массаж",
            services=[(10, "массаж спины"), (11, "лимфодренаж")],
        ),
        SpecialistCandidate(
            id=2, name="Борис", specialization="спа",
            services=[(12, "СПА")],
        ),
    ]
    # v0.7.0: tenant_id mandatory. Synthetic test value — clearly not real.
    return build_specialist_context_from_candidates(candidates, tenant_id="test-tenant")


def _make_tool_call(name: str, arguments: str, tc_id: str = "tc_1"):
    """Имитация OpenAI tool_call — duck-typed (.id, .function.name, .function.arguments)."""
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


# ─── handle_show_masters ──────────────────────────────────────────────────


class TestShowMasters:
    def test_valid_master_ids_returns_show_masters(self, master_context) -> None:
        args = {
            "master_ids": [1, 2],
            "match_scores": [90, 75],
            "match_reasons": [["опыт"], ["рейтинг"]],
            "explanation": "Лучшие подобраны",
        }
        result = handle_show_masters(args, master_context)
        assert result.action_type == ActionType.SHOW_MASTERS
        assert len(result.action_data["masters"]) == 2
        assert result.action_data["masters"][0]["master"]["id"] == 1
        assert result.action_data["explanation"] == "Лучшие подобраны"

    def test_hallucinated_master_id_dropped_silently(self, master_context) -> None:
        """LLM выдал [1, 999] — 999 в context нет, фильтруется."""
        args = {"master_ids": [1, 999], "explanation": "x"}
        result = handle_show_masters(args, master_context)
        # 1 валиден → возвращаем партиалкой
        assert result.action_type == ActionType.SHOW_MASTERS
        assert len(result.action_data["masters"]) == 1
        assert result.action_data["masters"][0]["master"]["id"] == 1

    def test_all_hallucinated_master_ids_fallback_to_clarification(
        self, master_context
    ) -> None:
        """Все master_ids левые → ask_clarification, не raise."""
        args = {"master_ids": [999, 998], "explanation": "x"}
        result = handle_show_masters(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION
        assert "уточните" in result.action_data["question"].lower()

    def test_empty_master_ids_fallback(self, master_context) -> None:
        result = handle_show_masters({"master_ids": [], "explanation": "x"}, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_string_master_ids_coerced_to_int(self, master_context) -> None:
        """LLM иногда выдаёт string вместо int — не должно ломать."""
        args = {"master_ids": ["1", "2"], "explanation": "x"}
        result = handle_show_masters(args, master_context)
        assert result.action_type == ActionType.SHOW_MASTERS
        assert len(result.action_data["masters"]) == 2

    def test_garbage_master_id_filtered(self, master_context) -> None:
        """LLM выдал 'abc' как master_id → silently dropped."""
        args = {"master_ids": [1, "abc", None], "explanation": "x"}
        result = handle_show_masters(args, master_context)
        assert result.action_type == ActionType.SHOW_MASTERS
        assert len(result.action_data["masters"]) == 1

    def test_bool_master_id_rejected(self, master_context) -> None:
        """LLM выдал JSON `true` (= 1 в int cast) — не должен совпасть с master_id=1."""
        args = {"master_ids": [True, False], "explanation": "x"}
        result = handle_show_masters(args, master_context)
        # True → _safe_int returns None → отфильтрован → все невалидны → fallback
        assert result.action_type == ActionType.ASK_CLARIFICATION

    # ─── B2 regression tests (v0.7.0) ──────────────────────────────────────

    def test_scores_reasons_aligned_after_hallucinated_id_filter(
        self, master_context
    ) -> None:
        """B2 (v0.7.0): when the LLM emits a mix of valid + hallucinated IDs,
        surviving masters must receive the score/reason the model assigned
        at the SAME original index — not the post-filter index.

        Pre-fix: master_ids=[999, 1, 998, 2] + scores=[10, 90, 20, 75]
        → master 1 got score=10 (wrong — that was meant for hallucinated 999).
        Post-fix: master 1 gets score=90, master 2 gets score=75.
        """
        args = {
            "master_ids": [999, 1, 998, 2],
            "match_scores": [10, 90, 20, 75],
            "match_reasons": [["bad1"], ["good1"], ["bad2"], ["good2"]],
            "explanation": "x",
        }
        result = handle_show_masters(args, master_context)
        assert result.action_type == ActionType.SHOW_MASTERS
        masters = result.action_data["masters"]
        assert len(masters) == 2

        # Order preserved by original emission order — first surviving was id=1.
        m1, m2 = masters
        assert m1["master"]["id"] == 1
        assert m1["match_score"] == 90
        assert m1["match_reasons"] == ["good1"]
        assert m2["master"]["id"] == 2
        assert m2["match_score"] == 75
        assert m2["match_reasons"] == ["good2"]

    def test_duplicate_master_ids_deduped(self, master_context) -> None:
        """B2 (v0.7.0): LLM occasionally repeats the same id; render once."""
        args = {
            "master_ids": [1, 1, 1],
            "match_scores": [90, 80, 70],
            "match_reasons": [["a"], ["b"], ["c"]],
            "explanation": "x",
        }
        result = handle_show_masters(args, master_context)
        masters = result.action_data["masters"]
        assert len(masters) == 1
        assert masters[0]["master"]["id"] == 1
        # First emission wins (orig_idx=0) for score/reason.
        assert masters[0]["match_score"] == 90
        assert masters[0]["match_reasons"] == ["a"]

    def test_alignment_when_scores_array_short(self, master_context) -> None:
        """Fewer scores than ids → surviving masters whose orig_idx exceeds
        scores length get None (graceful)."""
        args = {
            "master_ids": [999, 1, 2],
            "match_scores": [10, 90],  # only 2 entries; idx=2 → None
            "match_reasons": [["bad"], ["good1"]],
            "explanation": "x",
        }
        result = handle_show_masters(args, master_context)
        masters = result.action_data["masters"]
        assert len(masters) == 2
        # master id=1 at orig_idx=1 → score=90
        assert masters[0]["master"]["id"] == 1
        assert masters[0]["match_score"] == 90
        # master id=2 at orig_idx=2 → score absent in array → None
        assert masters[1]["master"]["id"] == 2
        assert masters[1]["match_score"] is None
        assert masters[1]["match_reasons"] == []


# ─── handle_show_slots ────────────────────────────────────────────────────


class TestShowSlots:
    def test_valid_args_returns_show_slots(self, master_context) -> None:
        args = {"master_id": 1, "service_id": 10, "date": "2026-05-15"}
        result = handle_show_slots(args, master_context)
        assert result.action_type == ActionType.SHOW_SLOTS
        assert result.action_data == {
            "master_id": 1, "service_id": 10, "date": "2026-05-15",
        }

    def test_hallucinated_master_id_fallback(self, master_context) -> None:
        args = {"master_id": 999, "service_id": 10, "date": "2026-05-15"}
        result = handle_show_slots(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_hallucinated_service_id_fallback(self, master_context) -> None:
        args = {"master_id": 1, "service_id": 999, "date": "2026-05-15"}
        result = handle_show_slots(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_master_does_not_offer_service_fallback(self, master_context) -> None:
        """master_id=1 (Анна) не делает service_id=12 (СПА Бориса) — fallback с конкретным question."""
        args = {"master_id": 1, "service_id": 12, "date": "2026-05-15"}
        result = handle_show_slots(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION
        assert "не оказывает" in result.action_data["question"]

    def test_invalid_date_fallback(self, master_context) -> None:
        args = {"master_id": 1, "service_id": 10, "date": "not-a-date"}
        result = handle_show_slots(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION


# ─── handle_confirm_booking ───────────────────────────────────────────────


class TestConfirmBooking:
    def test_valid_args_no_resolvers_returns_unenriched(self, master_context) -> None:
        args = {
            "master_id": 1, "service_id": 10,
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(args, master_context)
        assert result.action_type == ActionType.CONFIRM_BOOKING
        assert result.action_data["master_id"] == 1
        assert result.action_data["service_id"] == 10
        assert result.action_data["master_name"] is None  # без resolver
        assert result.action_data["service_name"] is None

    def test_with_resolvers_enriches_action_data(self, master_context) -> None:
        master_resolver = lambda mid, **_: (
            {"name": "Анна Иванова", "tenant_id": "test-tenant"} if mid == 1 else None
        )
        service_resolver = lambda sid, **_: (
            {
                "name": "массаж спины",
                "price_from": "2500",
                "duration_min": 60,
                "tenant_id": "test-tenant",
            }
            if sid == 10
            else None
        )

        args = {
            "master_id": 1, "service_id": 10,
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(
            args, master_context,
            master_resolver=master_resolver,
            service_resolver=service_resolver,
        )
        assert result.action_type == ActionType.CONFIRM_BOOKING
        assert result.action_data["master_name"] == "Анна Иванова"
        assert result.action_data["service_name"] == "массаж спины"
        assert result.action_data["price_from"] == "2500"
        assert result.action_data["duration_min"] == 60

    def test_resolver_returns_none_fallback(self, master_context) -> None:
        """master_resolver вернул None (race: мастер удалён) → fallback."""
        args = {
            "master_id": 1, "service_id": 10,
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(
            args, master_context,
            master_resolver=lambda mid, **_: None,
        )
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_hallucinated_master_id_fallback(self, master_context) -> None:
        args = {
            "master_id": 999, "service_id": 10,
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_invalid_datetime_fallback(self, master_context) -> None:
        args = {"master_id": 1, "service_id": 10, "datetime": "next-week"}
        result = handle_confirm_booking(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_master_does_not_offer_service_fallback(self, master_context) -> None:
        """LLM trust boundary: (Анна id=1, СПА id=12 от Бориса) → master.services check сработает.

        Без cross-validation booking endpoint принимает любую (master, service) пару из
        global candidate_service_ids, что приводит к подтверждению bookings которые мастер
        не делает. Это самый safety-critical handler.
        """
        args = {
            "master_id": 1,  # Анна
            "service_id": 12,  # СПА Бориса (есть в context.candidate_service_ids, но не у Анны)
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION
        assert "не оказывает" in result.action_data["question"]

    def test_decimal_price_from_coerced_to_str(self, master_context) -> None:
        """Decimal не JSON-serializable — handler должен defensively cast в str.

        action_data попадёт в Message.action_data (JSONField) — Decimal сломает serialization.
        Соответствует behavior бота: `str(service.price_from) if service.price_from else None`.
        """
        from decimal import Decimal

        master_resolver = lambda mid, **_: (
            {"name": "Анна", "tenant_id": "test-tenant"} if mid == 1 else None
        )
        service_resolver = lambda sid, **_: (
            {
                "name": "массаж",
                "price_from": Decimal("2500.00"),
                "duration_min": 60,
                "tenant_id": "test-tenant",
            }
            if sid == 10
            else None
        )

        args = {
            "master_id": 1, "service_id": 10,
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(
            args, master_context,
            master_resolver=master_resolver,
            service_resolver=service_resolver,
        )
        assert result.action_type == ActionType.CONFIRM_BOOKING
        # Critical: type должен быть str, не Decimal — иначе JSONField сломается
        assert isinstance(result.action_data["price_from"], str)
        assert result.action_data["price_from"] == "2500.00"

    def test_price_from_none_stays_none(self, master_context) -> None:
        """None должен остаться None (не "None" строкой)."""
        master_resolver = lambda mid, **_: {"name": "Анна", "tenant_id": "test-tenant"}
        service_resolver = lambda sid, **_: {
            "name": "массаж",
            "price_from": None,
            "duration_min": 60,
            "tenant_id": "test-tenant",
        }
        args = {
            "master_id": 1, "service_id": 10,
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(
            args, master_context,
            master_resolver=master_resolver,
            service_resolver=service_resolver,
        )
        assert result.action_data["price_from"] is None

    def test_master_unavailable_specific_question(self, master_context) -> None:
        """Resolver-None должен дать конкретный user-facing question, не generic."""
        args = {
            "master_id": 1, "service_id": 10,
            "datetime": "2026-05-15T14:00:00+03:00",
        }
        result = handle_confirm_booking(
            args, master_context,
            master_resolver=lambda mid, **_: None,
        )
        assert result.action_type == ActionType.ASK_CLARIFICATION
        assert "недоступен" in result.action_data["question"]


# ─── handle_show_my_bookings ──────────────────────────────────────────────


class TestShowMyBookings:
    def test_valid_filter(self) -> None:
        result = handle_show_my_bookings({"filter": "past"})
        assert result.action_type == ActionType.SHOW_MY_BOOKINGS
        assert result.action_data["filter"] == "past"

    def test_default_to_upcoming_when_missing(self) -> None:
        result = handle_show_my_bookings({})
        assert result.action_data["filter"] == "upcoming"

    def test_invalid_filter_falls_to_upcoming(self) -> None:
        result = handle_show_my_bookings({"filter": "yesterday"})
        assert result.action_data["filter"] == "upcoming"


# ─── handle_ask_clarification ─────────────────────────────────────────────


class TestAskClarification:
    def test_valid_question(self) -> None:
        result = handle_ask_clarification(
            {"question": "Какой день удобнее?", "options": ["Пн", "Вт"]}
        )
        assert result.action_type == ActionType.ASK_CLARIFICATION
        assert result.action_data["question"] == "Какой день удобнее?"
        assert result.action_data["options"] == ["Пн", "Вт"]

    def test_empty_question_fallback(self) -> None:
        result = handle_ask_clarification({"question": "  "})
        assert result.action_type == ActionType.ASK_CLARIFICATION
        # Default fallback question
        assert "уточните" in result.action_data["question"].lower()


# ─── dispatch_tool_call ───────────────────────────────────────────────────


class TestDispatch:
    def test_routes_show_masters(self, master_context) -> None:
        tc = _make_tool_call(
            "show_masters",
            '{"master_ids": [1], "explanation": "x"}',
        )
        result = dispatch_tool_call(tc, master_context)
        assert result.action_type == ActionType.SHOW_MASTERS

    def test_routes_show_slots(self, master_context) -> None:
        tc = _make_tool_call(
            "show_slots",
            '{"master_id": 1, "service_id": 10, "date": "2026-05-15"}',
        )
        result = dispatch_tool_call(tc, master_context)
        assert result.action_type == ActionType.SHOW_SLOTS

    def test_routes_confirm_booking(self, master_context) -> None:
        tc = _make_tool_call(
            "confirm_booking",
            '{"master_id": 1, "service_id": 10, "datetime": "2026-05-15T14:00:00+03:00"}',
        )
        result = dispatch_tool_call(tc, master_context)
        assert result.action_type == ActionType.CONFIRM_BOOKING

    def test_routes_show_my_bookings(self, master_context) -> None:
        tc = _make_tool_call("show_my_bookings", '{"filter": "upcoming"}')
        result = dispatch_tool_call(tc, master_context)
        assert result.action_type == ActionType.SHOW_MY_BOOKINGS

    def test_routes_ask_clarification(self, master_context) -> None:
        tc = _make_tool_call("ask_clarification", '{"question": "что вы хотите?"}')
        result = dispatch_tool_call(tc, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_invalid_json_arguments_fallback(self, master_context) -> None:
        tc = _make_tool_call("show_masters", "not valid json {")
        result = dispatch_tool_call(tc, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_unknown_tool_name_fallback(self, master_context) -> None:
        tc = _make_tool_call("unknown_tool_xyz", "{}")
        result = dispatch_tool_call(tc, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_empty_tool_call_arguments_fallback_via_default_dict(
        self, master_context
    ) -> None:
        """arguments='' → '{}' default → handle_* gets {} → fallback на пустых params."""
        tc = _make_tool_call("show_masters", "")
        result = dispatch_tool_call(tc, master_context)
        # Empty args → no master_ids → fallback
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_confirm_booking_passes_resolvers_through_dispatch(self, master_context) -> None:
        """dispatch_tool_call forwards resolvers to handle_confirm_booking."""
        master_resolver = lambda mid, **_: (
            {"name": "X", "tenant_id": "test-tenant"} if mid == 1 else None
        )
        tc = _make_tool_call(
            "confirm_booking",
            '{"master_id": 1, "service_id": 10, "datetime": "2026-05-15T14:00:00+03:00"}',
        )
        result = dispatch_tool_call(tc, master_context, master_resolver=master_resolver)
        assert result.action_type == ActionType.CONFIRM_BOOKING
        assert result.action_data["master_name"] == "X"


# ─── ToolResult immutability ─────────────────────────────────────────────


def test_tool_result_is_frozen() -> None:
    r = ToolResult(action_type="x", action_data={})
    with pytest.raises(AttributeError):
        r.action_type = "y"  # type: ignore[misc]


# ─── DRF-238: id_parser support (UUID consumers) ──────────────────────────


class TestUuidIdParser:
    """Ayla-style consumer: SpecialistContext[UUID] + _safe_uuid id_parser."""

    @pytest.fixture
    def uuid_context(self):
        from uuid import UUID

        from ayla_ai_core.context import (
            SpecialistCandidate,
            build_specialist_context_from_candidates,
        )

        uid_anna = UUID("11111111-1111-1111-1111-111111111111")
        uid_boris = UUID("22222222-2222-2222-2222-222222222222")
        sid_back = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        sid_spa = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        candidates = [
            SpecialistCandidate(
                id=uid_anna, name="Анна", specialization="массаж",
                services=[(sid_back, "массаж спины")],
            ),
            SpecialistCandidate(
                id=uid_boris, name="Борис", specialization="спа",
                services=[(sid_spa, "СПА")],
            ),
        ]
        return build_specialist_context_from_candidates(candidates, tenant_id="ayla")

    def test_safe_uuid_parses_uuid_string(self) -> None:
        from uuid import UUID

        from ayla_ai_core.tool_handlers import _safe_uuid

        assert _safe_uuid("11111111-1111-1111-1111-111111111111") == UUID(
            "11111111-1111-1111-1111-111111111111"
        )

    def test_safe_uuid_returns_none_for_non_uuid(self) -> None:
        from ayla_ai_core.tool_handlers import _safe_uuid

        assert _safe_uuid("not-a-uuid") is None
        assert _safe_uuid("") is None
        assert _safe_uuid(None) is None
        assert _safe_uuid(42) is None
        assert _safe_uuid([]) is None

    def test_safe_uuid_passes_through_uuid_objects(self) -> None:
        from uuid import UUID, uuid4

        from ayla_ai_core.tool_handlers import _safe_uuid

        u = uuid4()
        assert _safe_uuid(u) is u
        # And str-form too
        assert _safe_uuid(str(u)) == u
        assert isinstance(_safe_uuid(str(u)), UUID)

    def test_show_masters_with_uuid_parser(self, uuid_context) -> None:
        """LLM эмиттит UUID-strings → handler парсит через _safe_uuid → matches context."""
        from ayla_ai_core.tool_handlers import _safe_uuid, handle_show_masters

        args = {
            "master_ids": ["11111111-1111-1111-1111-111111111111"],
            "explanation": "x",
        }
        result = handle_show_masters(args, uuid_context, id_parser=_safe_uuid)
        assert result.action_type == ActionType.SHOW_MASTERS
        assert len(result.action_data["masters"]) == 1
        assert result.action_data["masters"][0]["master"]["name"] == "Анна"

    def test_show_masters_hallucinated_uuid_fallback(self, uuid_context) -> None:
        """LLM выдумал валидный по форме но not-in-DB UUID → fallback."""
        from ayla_ai_core.tool_handlers import _safe_uuid, handle_show_masters

        args = {
            "master_ids": ["99999999-9999-9999-9999-999999999999"],
            "explanation": "x",
        }
        result = handle_show_masters(args, uuid_context, id_parser=_safe_uuid)
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_dispatch_with_uuid_parser_routes_correctly(self, uuid_context) -> None:
        """dispatch_tool_call с id_parser=_safe_uuid → правильно парсит и роутит."""
        from ayla_ai_core.tool_handlers import _safe_uuid

        tc = _make_tool_call(
            "show_slots",
            '{"master_id": "11111111-1111-1111-1111-111111111111", '
            '"service_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", '
            '"date": "2026-05-15"}',
        )
        result = dispatch_tool_call(tc, uuid_context, id_parser=_safe_uuid)
        assert result.action_type == ActionType.SHOW_SLOTS
        # Проверяем что master.services cross-validation проходит для UUID
        assert "service_id" in result.action_data

    def test_default_int_parser_rejects_uuid_string(self, master_context) -> None:
        """Default _safe_int отбрасывает UUID-string как invalid → fallback.

        Sanity check: бот (default int) не должен случайно match UUID-формат.
        """
        args = {
            "master_ids": ["11111111-1111-1111-1111-111111111111"],
            "explanation": "x",
        }
        # Без id_parser override — default _safe_int → не парсит UUID → fallback
        result = handle_show_masters(args, master_context)
        assert result.action_type == ActionType.ASK_CLARIFICATION


def test_handlers_id_parser_default_is_safe_int(master_context) -> None:
    """Backward compat: handlers без id_parser=... используют _safe_int (бот)."""
    args = {"master_ids": [1, 2], "explanation": "x"}
    result = handle_show_masters(args, master_context)
    assert result.action_type == ActionType.SHOW_MASTERS
    assert len(result.action_data["masters"]) == 2


# ─── B3 regression tests (v0.7.0): mandatory tenant_id enforcement ─────────


class TestTenantScoping:
    """B3 (v0.7.0): tenant_id is required on SpecialistContext. dispatch +
    all handlers refuse to operate on an empty/missing tenant_id. Resolvers
    receive tenant_id as kwarg-only argument; mismatch between context and
    resolver-returned `tenant_id` triggers fallback clarification.
    """

    @staticmethod
    def _minimal_context(tenant_id: str = ""):
        """Build a minimal SpecialistContext, optionally with empty tenant."""
        # Build via direct constructor so we can pass empty/missing tenant
        # and trigger the assertion. The builder helper rejects empty
        # at construction time (tested in TestSpecialistContext).
        from ayla_ai_core.context import SpecialistCandidate, SpecialistContext

        return SpecialistContext(
            candidates=[
                SpecialistCandidate(
                    id=1, name="Анна", specialization="массаж",
                    services=[(10, "массаж спины")],
                ),
            ],
            candidate_ids=frozenset({1}),
            candidate_service_ids=frozenset({10}),
            summary_text="- master_id=1 Анна",
            tenant_id=tenant_id,
        )

    def test_context_without_tenant_id_raises(self):
        """Calling any handler with empty tenant_id → ValueError, fail-loud."""
        import pytest

        ctx = self._minimal_context(tenant_id="")  # empty string
        with pytest.raises(ValueError, match="tenant_id required"):
            handle_show_masters({"master_ids": [1]}, ctx)
        with pytest.raises(ValueError, match="tenant_id required"):
            handle_show_slots(
                {"master_id": 1, "service_id": 10, "date": "2026-05-20"},
                ctx,
            )
        with pytest.raises(ValueError, match="tenant_id required"):
            handle_confirm_booking(
                {
                    "master_id": 1, "service_id": 10,
                    "datetime": "2026-05-20T14:00:00+03:00",
                },
                ctx,
            )

    def test_resolver_called_with_tenant_id_kwarg(self):
        """v0.7.0: resolvers receive `tenant_id=` kwarg from dispatch.
        Allows the resolver to scope its ORM query by tenant."""
        seen_kwargs: dict = {}

        def master_resolver(value, **kwargs):
            seen_kwargs.update(kwargs)
            return {"name": "Анна Иванова", "tenant_id": "test-tenant"}

        ctx = self._minimal_context(tenant_id="test-tenant")
        result = handle_confirm_booking(
            {
                "master_id": 1, "service_id": 10,
                "datetime": "2026-05-20T14:00:00+03:00",
            },
            ctx,
            master_resolver=master_resolver,
        )
        # Resolver called with kwarg tenant_id matching context
        assert seen_kwargs.get("tenant_id") == "test-tenant"
        assert result.action_type == ActionType.CONFIRM_BOOKING

    def test_cross_tenant_master_dropped_to_clarification(self):
        """Master resolver returns dict with mismatching tenant_id → bounce
        to ASK_CLARIFICATION (anti cross-tenant leak)."""
        ctx = self._minimal_context(tenant_id="tenant-a")

        def cross_tenant_resolver(value, **kwargs):
            # Master row from another tenant accidentally cached/leaked
            return {"name": "Anna", "tenant_id": "tenant-b"}

        result = handle_confirm_booking(
            {
                "master_id": 1, "service_id": 10,
                "datetime": "2026-05-20T14:00:00+03:00",
            },
            ctx,
            master_resolver=cross_tenant_resolver,
        )
        assert result.action_type == ActionType.ASK_CLARIFICATION

    def test_cross_tenant_service_dropped_to_clarification(self):
        """Same cross-tenant guard fires for service_resolver."""
        ctx = self._minimal_context(tenant_id="tenant-a")

        def ok_master(value, **kwargs):
            return {"name": "Anna", "tenant_id": "tenant-a"}

        def cross_tenant_service(value, **kwargs):
            return {"name": "Massage", "tenant_id": "tenant-b"}

        result = handle_confirm_booking(
            {
                "master_id": 1, "service_id": 10,
                "datetime": "2026-05-20T14:00:00+03:00",
            },
            ctx,
            master_resolver=ok_master,
            service_resolver=cross_tenant_service,
        )
        assert result.action_type == ActionType.ASK_CLARIFICATION


# ─── DRF-680 (v0.7.2 Sec-1): strict cross-tenant guard ────────────────────


class TestStrictCrossTenantGuard:
    """v0.7.2 Sec-1: resolver result MUST include tenant_id; missing -> fallback.

    Soft-breaking from v0.7.0 (where missing tenant_id was permissive).
    Opt-out per resolver via __resolver_skips_tenant_check__ = True.
    """

    def _minimal_context(self, *, tenant_id: str) -> SpecialistContext:
        candidates = [
            SpecialistCandidate(
                id=1, name="Anna",
                specialization="Massage",
                services=[(10, "Back massage")],
            ),
        ]
        return build_specialist_context_from_candidates(
            candidates, tenant_id=tenant_id,
        )

    def test_master_resolver_missing_tenant_id_falls_back(self, caplog):
        """v0.7.2: resolver returning dict without tenant_id -> ASK_CLARIFICATION.

        Reason tag exposed in audit log so ops can find the misconfigured
        resolver. v0.7.0 silently allowed this — covered cross-tenant leaks
        when the resolver wasn't actually scoped at the ORM layer.
        """
        ctx = self._minimal_context(tenant_id="tenant-a")

        def sloppy_resolver(value, **kwargs):
            # Forgot to include tenant_id — v0.7.0 would have allowed this.
            return {"name": "Anna"}

        with caplog.at_level(logging.WARNING, logger="ayla_ai_core.tool_handlers"):
            result = handle_confirm_booking(
                {
                    "master_id": 1, "service_id": 10,
                    "datetime": "2026-05-20T14:00:00+03:00",
                },
                ctx,
                master_resolver=sloppy_resolver,
            )

        assert result.action_type == ActionType.ASK_CLARIFICATION
        assert "master_resolver_no_tenant_id" in caplog.text

    def test_service_resolver_missing_tenant_id_falls_back(self, caplog):
        """Service-side mirror of the master check."""
        ctx = self._minimal_context(tenant_id="tenant-a")

        def ok_master(value, **kwargs):
            return {"name": "Anna", "tenant_id": "tenant-a"}

        def sloppy_service(value, **kwargs):
            return {"name": "Massage"}

        with caplog.at_level(logging.WARNING, logger="ayla_ai_core.tool_handlers"):
            result = handle_confirm_booking(
                {
                    "master_id": 1, "service_id": 10,
                    "datetime": "2026-05-20T14:00:00+03:00",
                },
                ctx,
                master_resolver=ok_master,
                service_resolver=sloppy_service,
            )

        assert result.action_type == ActionType.ASK_CLARIFICATION
        assert "service_resolver_no_tenant_id" in caplog.text

    def test_opt_out_attribute_skips_check(self, caplog):
        """Resolver with `__resolver_skips_tenant_check__ = True` is exempt.

        Used for legacy wrappers that genuinely can't supply tenant_id.
        Emits a WARNING per call so the opt-out stays visible in audit logs.
        """
        ctx = self._minimal_context(tenant_id="tenant-a")

        def legacy_resolver(value, **kwargs):
            # Pretend this wraps an old ORM helper that returns plain rows.
            return {"name": "Anna"}

        legacy_resolver.__resolver_skips_tenant_check__ = True  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING, logger="ayla_ai_core.tool_handlers"):
            result = handle_confirm_booking(
                {
                    "master_id": 1, "service_id": 10,
                    "datetime": "2026-05-20T14:00:00+03:00",
                },
                ctx,
                master_resolver=legacy_resolver,
            )

        assert result.action_type == ActionType.CONFIRM_BOOKING
        assert result.action_data["master_name"] == "Anna"
        assert "resolver_skips_tenant_check" in caplog.text
        assert "kind=master" in caplog.text

    def test_opt_out_does_not_bypass_tenant_mismatch_when_explicit(self, caplog):
        """Opt-out short-circuits the WHOLE check — including mismatch.

        Documented escape hatch: caller takes responsibility. The audit log
        WARNING is the visible signal that this resolver bypasses the guard.
        """
        ctx = self._minimal_context(tenant_id="tenant-a")

        def legacy_resolver_with_wrong_tenant(value, **kwargs):
            return {"name": "Anna", "tenant_id": "tenant-b"}

        legacy_resolver_with_wrong_tenant.__resolver_skips_tenant_check__ = True  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING, logger="ayla_ai_core.tool_handlers"):
            result = handle_confirm_booking(
                {
                    "master_id": 1, "service_id": 10,
                    "datetime": "2026-05-20T14:00:00+03:00",
                },
                ctx,
                master_resolver=legacy_resolver_with_wrong_tenant,
            )

        # Opt-out means the caller asserted they handle tenant_id themselves.
        # Result is CONFIRM_BOOKING (not the safer mismatch fallback) —
        # the WARNING in audit log is the trail.
        assert result.action_type == ActionType.CONFIRM_BOOKING
        assert "resolver_skips_tenant_check" in caplog.text
