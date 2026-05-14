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

from ayla_ai_core.composer import PromptComposer
from ayla_ai_core.context import (
    ID_T,
    CandidateContext,
    ItemT,
    SpecialistCandidate,
    SpecialistContext,
    build_specialist_context_from_candidates,
    render_summary_text,
)
from ayla_ai_core.observability import (
    ReplayDeterminismError,
    TenantContextFilter,
    current_frozen_now,
    current_tenant_id,
    reset_tenant_id,
    scope_frozen_now,
    scope_tenant_id,
    set_tenant_id,
)
from ayla_ai_core.orchestrator import (
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_HISTORY_TOKEN_BUDGET,
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
from ayla_ai_core.providers import (
    AnthropicCompletionAdapter,
    CompletionAdapter,
    OpenAIPassthroughAdapter,
)
from ayla_ai_core.tool_handlers import (
    MasterResolver,
    ServiceResolver,
    ToolResult,
    dispatch_tool_call,
    parse_int,
    parse_uuid,
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

__version__ = "0.8.1"

__all__ = [
    # Tool definitions (factory + default int constants)
    "ASK_CLARIFICATION",
    # Brand voice configs (DRF-239)
    "AYLA_MARKETPLACE_VOICE",
    "CONFIRM_BOOKING",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_HISTORY_TOKEN_BUDGET",
    "DEFAULT_MODEL_NAME",
    "FORMULA_TELA_VOICE",
    "ID_T",
    "SHOW_MASTERS",
    "SHOW_MY_BOOKINGS",
    "SHOW_SLOTS",
    "TOOL_DEFINITIONS",
    "AIConcierge",
    "ActionType",
    # v0.8.0 (Arch-5 / DRF-689): pluggable provider adapters
    "AnthropicCompletionAdapter",
    "BrandVoiceConfig",
    # v0.8.0 (Arch-1 / DRF-685): structural Protocol over context shapes
    "CandidateContext",
    "ChatResponseDTO",
    "CompletionAdapter",
    "ConversationStore",
    "Example",
    "ItemT",
    # Backward compat (deprecated, для бота до DRF-243)
    "MasterResolver",
    "MessageRole",
    "OpenAIPassthroughAdapter",
    # v0.8.0 (Arch-2 / DRF-686): pluggable section-based prompt builder
    "PromptComposer",
    # v0.7.3 observability surface (re-exported in v0.7.4 — DRF-681..684 follow-up)
    "ReplayDeterminismError",
    "ServiceResolver",
    # Generic context (preferred)
    "SpecialistCandidate",
    "SpecialistContext",
    "TenantContextFilter",
    # DI hook for custom wire-format consumers (DRF-241 / 0.6.0)
    "ToolDispatcher",
    "ToolResult",
    "__version__",
    "build_specialist_context_from_candidates",
    "build_tool_definitions",
    "current_frozen_now",
    "current_tenant_id",
    "dispatch_tool_call",
    # v0.8.0 (Arch-6): canonical ID parsers. _safe_int / _safe_uuid
    # underscored aliases reachable via __getattr__ below with
    # DeprecationWarning — scheduled removal in v0.9.0.
    "parse_int",
    "parse_uuid",
    "render_summary_text",
    "render_system_prompt",
    "reset_tenant_id",
    "scope_frozen_now",
    "scope_tenant_id",
    "set_tenant_id",
]


# PEP 562 module-level __getattr__ — fires only for attributes NOT defined
# at module scope. v0.8.0 (Arch-6): keep `_safe_int` / `_safe_uuid` accessible
# via `from ayla_ai_core import _safe_int` but emit a DeprecationWarning on
# every import so consumers see the rename before v0.9.0 removal.
_DEPRECATED_ID_PARSER_ALIASES = {
    "_safe_int": "parse_int",
    "_safe_uuid": "parse_uuid",
}


def __getattr__(name: str) -> object:
    """Fallback attribute lookup for deprecated v0.8.x aliases.

    Pre-v0.9.0 the package keeps ``_safe_int`` / ``_safe_uuid`` accessible
    via this hook but emits a ``DeprecationWarning`` so consumers can
    migrate to ``parse_int`` / ``parse_uuid`` before the next minor.
    """
    if name in _DEPRECATED_ID_PARSER_ALIASES:
        import warnings

        new_name = _DEPRECATED_ID_PARSER_ALIASES[name]
        warnings.warn(
            f"`{name}` is deprecated since v0.8.0; use `{new_name}` instead. "
            f"The alias will be removed in v0.9.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ayla_ai_core import tool_handlers
        return getattr(tool_handlers, name)
    raise AttributeError(f"module 'ayla_ai_core' has no attribute {name!r}")
