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

from types import SimpleNamespace

import pytest

from ayla_ai_core.context import MasterCandidate, build_master_context_from_candidates
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
        MasterCandidate(
            id=1, name="Анна", specialization="массаж",
            services=[(10, "массаж спины"), (11, "лимфодренаж")],
        ),
        MasterCandidate(
            id=2, name="Борис", specialization="спа",
            services=[(12, "СПА")],
        ),
    ]
    return build_master_context_from_candidates(candidates)


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
        master_resolver = lambda mid: {"name": "Анна Иванова"} if mid == 1 else None
        service_resolver = lambda sid: {
            "name": "массаж спины", "price_from": "2500", "duration_min": 60,
        } if sid == 10 else None

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
            master_resolver=lambda mid: None,
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
        master_resolver = lambda mid: {"name": "X"} if mid == 1 else None
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
