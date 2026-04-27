"""ayla-ai-core — shared AI orchestration core for Ayla and Formula tela bot.

Public API:
- AIConcierge — main orchestrator
- ChatResponseDTO — return type для send_message
- ConversationStore — Protocol для persistence backend (структурный typing)
- MessageRole — wire-format для Message.role
- MasterContext / MasterCandidate — Top-N кандидаты с anti-hallucination IDs
- TOOL_DEFINITIONS / ActionType — 5 OpenAI tools (show_masters, show_slots,
  confirm_booking, show_my_bookings, ask_clarification)
- dispatch_tool_call / handle_* — handlers с anti-hallucination filter

Извлечено из `mysite/maxbot/` (production-tested 30+ days в Формуле тела).
См. `docs/BOT_CODE_AUDIT_2026-04.md` (in djangoproject) для extraction plan
и `docs/PRODUCT_AUDIT_2026-04.md` для strategic context.
"""
from __future__ import annotations

from ayla_ai_core.context import (
    MasterCandidate,
    MasterContext,
    build_master_context_from_candidates,
    render_summary_text,
)
from ayla_ai_core.orchestrator import (
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_MODEL_NAME,
    AIConcierge,
    ChatResponseDTO,
    ConversationStore,
    MessageRole,
)
from ayla_ai_core.tool_handlers import (
    MasterResolver,
    ServiceResolver,
    ToolResult,
    dispatch_tool_call,
    handle_ask_clarification,
    handle_confirm_booking,
    handle_show_masters,
    handle_show_my_bookings,
    handle_show_slots,
)
from ayla_ai_core.tools import (
    ASK_CLARIFICATION,
    CONFIRM_BOOKING,
    SHOW_MASTERS,
    SHOW_MY_BOOKINGS,
    SHOW_SLOTS,
    TOOL_DEFINITIONS,
    ActionType,
)

__version__ = "0.2.0"

__all__ = [
    "ASK_CLARIFICATION",
    "CONFIRM_BOOKING",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MODEL_NAME",
    "SHOW_MASTERS",
    "SHOW_MY_BOOKINGS",
    "SHOW_SLOTS",
    "TOOL_DEFINITIONS",
    # orchestrator
    "AIConcierge",
    # tools
    "ActionType",
    "ChatResponseDTO",
    "ConversationStore",
    # context
    "MasterCandidate",
    "MasterContext",
    "MasterResolver",
    "MessageRole",
    "ServiceResolver",
    # tool_handlers
    "ToolResult",
    "__version__",
    "build_master_context_from_candidates",
    "dispatch_tool_call",
    "handle_ask_clarification",
    "handle_confirm_booking",
    "handle_show_masters",
    "handle_show_my_bookings",
    "handle_show_slots",
    "render_summary_text",
]
