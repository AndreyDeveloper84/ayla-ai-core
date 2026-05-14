"""v0.8.1 (DRF-695 preview): public API surface snapshot test.

The `__all__` of `ayla_ai_core` is pinned here so accidental adds/removes
fail CI. Use this as the **drift gate** for every PR. v1.0.0 will promote
this snapshot to a hard freeze (per :doc:`LTS_POLICY.md`); v0.8.x still
allows additions and removals per the pre-1.0 caveat, but each requires
a deliberate update to ``EXPECTED_PUBLIC_API`` below + a CHANGELOG entry.

Adding or removing a symbol from `__all__` requires:

1. A deliberate CHANGELOG entry under the appropriate version
2. Updating this test (and only this test pins it)
3. Post-v1.0.0: REMOVING a symbol is a v2.0 hard break per LTS_POLICY.md

The list is sorted to match the file's `__all__` ordering convention,
making diffs reviewable.
"""
from __future__ import annotations

# Current public API snapshot. Add a symbol here when adding to __all__;
# remove with a CHANGELOG break-note. Post-v1.0.0 this becomes immutable
# minus deliberate v2.0 majors.
EXPECTED_PUBLIC_API: frozenset[str] = frozenset({
    # Tool definitions
    "ASK_CLARIFICATION",
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
    # Core types
    "AIConcierge",
    "ActionType",
    "AnthropicCompletionAdapter",
    "BrandVoiceConfig",
    "CandidateContext",
    "ChatResponseDTO",
    "CompletionAdapter",
    "ConversationStore",
    "Example",
    "ItemT",
    "MasterResolver",
    "MessageRole",
    "OpenAIPassthroughAdapter",
    "PromptComposer",
    "ReplayDeterminismError",
    "ServiceResolver",
    "SpecialistCandidate",
    "SpecialistContext",
    "TenantContextFilter",
    "ToolDispatcher",
    "ToolResult",
    "__version__",
    # Functions
    "build_specialist_context_from_candidates",
    "build_tool_definitions",
    "current_frozen_now",
    "current_tenant_id",
    "dispatch_tool_call",
    "parse_int",
    "parse_uuid",
    "render_summary_text",
    "render_system_prompt",
    "reset_tenant_id",
    "scope_frozen_now",
    "scope_tenant_id",
    "set_tenant_id",
})


def test_public_api_matches_expected_snapshot() -> None:
    """`__all__` must equal :data:`EXPECTED_PUBLIC_API` exactly.

    Failure modes:

    - **Added symbol**: update :data:`EXPECTED_PUBLIC_API` and add a
      CHANGELOG entry under the version that introduces it. Additions
      in v0.8.x are allowed (pre-1.0 caveat); post-1.0 they're semver
      minor.
    - **Removed symbol**: update CHANGELOG with a break-note. Pre-v1.0
      this is allowed under the documented caveat; post-v1.0 it's a
      v2.0 hard break per :doc:`LTS_POLICY.md`.
    """
    import ayla_ai_core

    actual = frozenset(ayla_ai_core.__all__)
    added = actual - EXPECTED_PUBLIC_API
    removed = EXPECTED_PUBLIC_API - actual

    assert not added, (
        f"API addition detected: {sorted(added)}. "
        "Add to EXPECTED_PUBLIC_API and CHANGELOG.md, then re-run."
    )
    assert not removed, (
        f"API removal detected: {sorted(removed)}. "
        "Update CHANGELOG.md break-note. "
        "Post-v1.0.0 this requires a major-version bump per LTS_POLICY.md."
    )


def test_every_public_symbol_is_actually_exported() -> None:
    """Sanity: every name in `__all__` resolves to an attribute on the
    package. Stops a typo in __all__ from quietly creating an unimportable
    symbol.
    """
    import ayla_ai_core

    missing: list[str] = []
    for name in ayla_ai_core.__all__:
        if not hasattr(ayla_ai_core, name):
            missing.append(name)
    assert not missing, (
        f"`__all__` references undefined attributes: {missing}. "
        "Either remove from __all__ or add the missing import in __init__.py."
    )


def test_every_public_symbol_has_a_docstring_or_is_data() -> None:
    """Every CALLABLE / CLASS in `__all__` carries a docstring. Pure
    data constants (`TOOL_DEFINITIONS`, `__version__`, `ID_T`) are
    exempt because they don't have a `__doc__` slot in the Python sense.
    Will become strict via ``ruff`` D-rules in v1.0.0 / Stab-1.

    Enforcing this here (rather than via ruff D-rules globally) keeps the
    check scoped to the **public** surface; internals can stay terse.
    """
    import ayla_ai_core

    # Constants + TypeVars don't carry independent docstrings.
    data_or_typevar = {
        "_safe_int",  # v0.8.x deprecated alias — same object as parse_int
        "_safe_uuid",  # v0.8.x deprecated alias — same object as parse_uuid
        "ASK_CLARIFICATION",
        "AYLA_MARKETPLACE_VOICE",
        "CONFIRM_BOOKING",
        "DEFAULT_HISTORY_LIMIT",
        "DEFAULT_HISTORY_TOKEN_BUDGET",
        "DEFAULT_MODEL_NAME",
        "FORMULA_TELA_VOICE",
        "ID_T",
        "ItemT",
        "SHOW_MASTERS",
        "SHOW_MY_BOOKINGS",
        "SHOW_SLOTS",
        "TOOL_DEFINITIONS",
        "MasterResolver",
        "ServiceResolver",
        "ToolDispatcher",
        "MessageRole",
        "__version__",
    }

    missing_docs: list[str] = []
    for name in ayla_ai_core.__all__:
        if name in data_or_typevar:
            continue
        obj = getattr(ayla_ai_core, name)
        doc = getattr(obj, "__doc__", None)
        if not doc or not doc.strip():
            missing_docs.append(name)
    assert not missing_docs, (
        f"Public API symbols missing docstring: {missing_docs}. "
        "v1.0.0 / Stab-1 requires every public class/function to have "
        "at least a one-paragraph docstring."
    )
