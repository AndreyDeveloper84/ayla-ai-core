"""ayla-ai-core — shared AI orchestration core for Ayla and Formula tela bot.

Public API (semver-stable):
- AIConcierge — main orchestrator
- ChatResponseDTO — return type для send_message
- ConversationStore — Protocol для persistence backend (структурный typing)
- MessageRole — wire-format для Message.role
- SpecialistContext / SpecialistCandidate — Generic[ID_T] (int | UUID)
  Top-N кандидаты с anti-hallucination IDs + multi-tenant scope (DRF-238)
- TOOL_DEFINITIONS — default int IDs (бот). Для UUID:
  `build_tool_definitions("string")` (DRF-238)
- ActionType — 5 wire-format constants (show_masters, show_slots,
  confirm_booking, show_my_bookings, ask_clarification)
- dispatch_tool_call — главный seam для tool_call routing с anti-hallucination
- ToolResult — return type для dispatch_tool_call
- _safe_int / _safe_uuid — id parsers (DRF-238)
- MasterContext / MasterCandidate — DEPRECATED aliases на SpecialistContext[int].
  Бот использует до DRF-243 миграции; после удалить.

Internal helpers (handle_*) НЕ экспортируются — caller'ы используют
dispatch_tool_call. Прямой импорт остаётся возможным
(`from ayla_ai_core.tool_handlers import handle_show_masters`) — это слабый
contract для тестов и редкой интеграции, не часть semver-promise.

Извлечено из `mysite/maxbot/` (production-tested 30+ days в Формуле тела) в DRF-237.
DRF-238 — generic over ID type + multi-tenant.
См. `docs/BOT_CODE_AUDIT_2026-04.md` (в djangoproject) для extraction plan.

Roadmap:
- 0.4.0 (DRF-238, this release): SpecialistContext[ID_T] + tenant_id + id_parser.
  Pre-1.0 — minor releases могут быть breaking.
- 0.5.0 (DRF-239): BrandVoiceConfig + tenant-aware render_system_prompt.
- 1.0.0: API стабилизирован, бот мигрирован (DRF-243), Ayla M4-pilot live.
"""
from __future__ import annotations

from ayla_ai_core.context import (
    ID_T,
    MasterCandidate,
    MasterContext,
    SpecialistCandidate,
    SpecialistContext,
    build_master_context_from_candidates,
    build_specialist_context_from_candidates,
    render_summary_text,
)
from ayla_ai_core.orchestrator import (
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_MODEL_NAME,
    AIConcierge,
    ChatResponseDTO,
    ConversationStore,
    MessageRole,
    ToolDispatcher,
)
from ayla_ai_core.prompts import (
    AYLA_MARKETPLACE_VOICE,
    FORMULA_TELA_VOICE,
    BrandVoiceConfig,
    Example,
    render_system_prompt,
)
from ayla_ai_core.tool_handlers import (
    MasterResolver,
    ServiceResolver,
    ToolResult,
    _safe_int,
    _safe_uuid,
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
    build_tool_definitions,
)

__version__ = "0.7.2"

__all__ = [
    # Tool definitions (factory + default int constants)
    "ASK_CLARIFICATION",
    # Brand voice configs (DRF-239)
    "AYLA_MARKETPLACE_VOICE",
    "CONFIRM_BOOKING",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MODEL_NAME",
    "FORMULA_TELA_VOICE",
    "ID_T",
    "SHOW_MASTERS",
    "SHOW_MY_BOOKINGS",
    "SHOW_SLOTS",
    "TOOL_DEFINITIONS",
    "AIConcierge",
    "ActionType",
    "BrandVoiceConfig",
    "ChatResponseDTO",
    "ConversationStore",
    "Example",
    # Backward compat (deprecated, для бота до DRF-243)
    "MasterCandidate",
    "MasterContext",
    "MasterResolver",
    "MessageRole",
    "ServiceResolver",
    # Generic context (preferred)
    "SpecialistCandidate",
    "SpecialistContext",
    # DI hook for custom wire-format consumers (DRF-241 / 0.6.0)
    "ToolDispatcher",
    "ToolResult",
    "__version__",
    "_safe_int",
    "_safe_uuid",
    "build_master_context_from_candidates",
    "build_specialist_context_from_candidates",
    "build_tool_definitions",
    "dispatch_tool_call",
    "render_summary_text",
    "render_system_prompt",
]
