"""ayla-ai-core — shared AI orchestration core for Ayla and Formula tela bot.

Public API (semver-stable):
- AIConcierge — main orchestrator
- ChatResponseDTO — return type для send_message
- ConversationStore — Protocol для persistence backend (структурный typing)
- MessageRole — wire-format для Message.role
- MasterContext / MasterCandidate — Top-N кандидаты с anti-hallucination IDs
- TOOL_DEFINITIONS / ActionType — 5 OpenAI tools (show_masters, show_slots,
  confirm_booking, show_my_bookings, ask_clarification)
- dispatch_tool_call — главный seam для tool_call routing с anti-hallucination
- ToolResult — return type для dispatch_tool_call

Internal helpers (handle_*) НЕ экспортируются — caller'ы используют
dispatch_tool_call. Прямой импорт остаётся возможным
(`from ayla_ai_core.tool_handlers import handle_show_masters`) — это слабый
contract для тестов и редкой интеграции, не часть semver-promise.

Извлечено из `mysite/maxbot/` (production-tested 30+ days в Формуле тела) в DRF-237.
См. `docs/BOT_CODE_AUDIT_2026-04.md` (в djangoproject) для extraction plan
и `docs/PRODUCT_AUDIT_2026-04.md` для strategic context.

Roadmap:
- 0.3.0 (DRF-238): generic ID-type (int → UUID) — breaking change на TOOL_DEFINITIONS
  и MasterCandidate. Pre-1.0 минорные релизы могут быть breaking.
- 0.4.0 (DRF-239): BrandVoiceConfig + tenant-aware render_system_prompt.
- 1.0.0: API стабилизирован, бот мигрирован (DRF-243), Ayla M4-pilot live.
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

__version__ = "0.3.0"

__all__ = [
    "ASK_CLARIFICATION",
    "CONFIRM_BOOKING",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MODEL_NAME",
    "SHOW_MASTERS",
    "SHOW_MY_BOOKINGS",
    "SHOW_SLOTS",
    "TOOL_DEFINITIONS",
    "AIConcierge",
    "ActionType",
    "ChatResponseDTO",
    "ConversationStore",
    "MasterCandidate",
    "MasterContext",
    "MasterResolver",
    "MessageRole",
    "ServiceResolver",
    "ToolResult",
    "__version__",
    "build_master_context_from_candidates",
    "dispatch_tool_call",
    "render_summary_text",
]
