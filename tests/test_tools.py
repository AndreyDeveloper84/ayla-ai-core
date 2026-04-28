"""Tests для tools.py — TOOL_DEFINITIONS schema integrity + ActionType wire-format."""
from __future__ import annotations

from ayla_ai_core.tools import (
    ASK_CLARIFICATION,
    CONFIRM_BOOKING,
    SHOW_MASTERS,
    SHOW_MY_BOOKINGS,
    SHOW_SLOTS,
    TOOL_DEFINITIONS,
    ActionType,
)


def test_tool_definitions_has_5_tools() -> None:
    assert len(TOOL_DEFINITIONS) == 5


def test_tool_definitions_match_action_type_names() -> None:
    """function.name === ActionType constant — wire-format invariant."""
    names = {td["function"]["name"] for td in TOOL_DEFINITIONS}
    assert names == ActionType.ALL_MVP


def test_each_tool_has_required_jsonschema_fields() -> None:
    for td in TOOL_DEFINITIONS:
        assert td["type"] == "function"
        fn = td["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_show_masters_requires_master_ids_and_explanation() -> None:
    required = SHOW_MASTERS["function"]["parameters"]["required"]
    assert "master_ids" in required
    assert "explanation" in required


def test_show_slots_requires_master_service_date() -> None:
    required = SHOW_SLOTS["function"]["parameters"]["required"]
    assert set(required) == {"master_id", "service_id", "date"}


def test_confirm_booking_requires_master_service_datetime() -> None:
    required = CONFIRM_BOOKING["function"]["parameters"]["required"]
    assert set(required) == {"master_id", "service_id", "datetime"}


def test_show_my_bookings_filter_is_enum() -> None:
    props = SHOW_MY_BOOKINGS["function"]["parameters"]["properties"]
    assert props["filter"]["enum"] == ["upcoming", "past", "all"]


def test_ask_clarification_requires_question() -> None:
    required = ASK_CLARIFICATION["function"]["parameters"]["required"]
    assert "question" in required


def test_action_type_constants_string_values() -> None:
    """Action types — strings (для Django TextChoices совместимости)."""
    assert ActionType.SHOW_MASTERS == "show_masters"
    assert ActionType.SHOW_SLOTS == "show_slots"
    assert ActionType.CONFIRM_BOOKING == "confirm_booking"
    assert ActionType.SHOW_MY_BOOKINGS == "show_my_bookings"
    assert ActionType.ASK_CLARIFICATION == "ask_clarification"


def test_action_type_all_mvp_is_frozenset() -> None:
    assert isinstance(ActionType.ALL_MVP, frozenset)
    assert len(ActionType.ALL_MVP) == 5


def test_handle_helpers_not_in_public_api() -> None:
    """handle_* helpers — internal, semver-неустойчивы. Caller'ы используют dispatch_tool_call.

    Прямой импорт `from ayla_ai_core.tool_handlers import handle_X` остаётся
    возможным для тестов и редкой интеграции — это слабый contract, не часть API.
    """
    import ayla_ai_core
    public = set(ayla_ai_core.__all__)
    leaked_helpers = {n for n in public if n.startswith("handle_")}
    assert leaked_helpers == set(), f"handle_* leaked into public API: {leaked_helpers}"


def test_dispatch_tool_call_is_public_seam() -> None:
    """dispatch_tool_call — единственный public seam для tool routing."""
    import ayla_ai_core
    assert "dispatch_tool_call" in ayla_ai_core.__all__
