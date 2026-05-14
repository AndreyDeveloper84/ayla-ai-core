# Changelog

All notable changes to `ayla-ai-core` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with **pre-1.0 caveats** — minor releases may be breaking. Each release ships a
migration guide. Consumers pin by SHA (not tag — tags are force-pushable).

---

## [Unreleased]

## [0.8.0] — 2026-05-14 (PR-3 of 3 — final)

Third and final PR of the v0.8.0 rollout. **Hard-breaking** in two
narrow places: the v0.4-era Master\* aliases are removed and Django
moves from a runtime dependency to an optional `[django]` extra.
Consumers using the canonical `Specialist*` names + installing the
library with `ayla-ai-core[django]` see zero migration cost.

### Removed

- **Master\* aliases** (Arch-4 / DRF-688):
  - `MasterCandidate` — use `SpecialistCandidate[int]`
  - `MasterContext` — use `SpecialistContext[int]`
  - `build_master_context_from_candidates(...)` — use
    `build_specialist_context_from_candidates(...)`
  
  These were marked DEPRECATED since v0.5+ but kept for the FROZEN
  `mysite/maxbot/` consumer (which is pinned to v0.6.0 and never sees
  v0.8.0). New regression tests in
  `tests/test_candidate_context.py::TestArch4DeprecatedAliasesRemoved`
  pin the removal so accidental re-introduction fails CI.
- **Implicit Django runtime dependency** (Arch-3 / DRF-687):
  - `pyproject.toml` no longer lists `django>=5.2,<6.0` in
    `[project.dependencies]`. Moved to the new `[django]` optional
    extra so plain `pip install ayla-ai-core` produces a Python-only
    library with no Django pull.
  - Bot consumer (`mysite/maxbot/`) + ai-bot-platform pin via
    `ayla-ai-core[django] @ git+...` — for them, nothing changes.
  - New regression test
    `tests/test_candidate_context.py::TestArch3DjangoExtra::test_no_django_import_in_src`
    walks every `.py` file in `src/ayla_ai_core/` and asserts no
    `import django` / `from django` lines exist.

### Migration cookbook (single PR for typical consumer)

```toml
# pyproject.toml — switch to the [django] extra (one-line change)
ai-core = [
-    "ayla-ai-core @ git+https://github.com/...@<v0.7.4-SHA>",
+    "ayla-ai-core[django] @ git+https://github.com/...@<v0.8.0-SHA>",
]
```

```python
# Any code still referencing the Master* aliases — sed-replace:
- from ayla_ai_core import MasterCandidate, MasterContext, build_master_context_from_candidates
+ from ayla_ai_core import (
+     SpecialistCandidate,
+     SpecialistContext,
+     build_specialist_context_from_candidates,
+ )
```

That's it. Suite: **218/218 pass** (was 214 in rc2; +4 regression tests
covering Arch-3 + Arch-4 contracts).

### What v0.8.0 delivers end-to-end (rc1 + rc2 + final)

- **Arch-5** (rc1): pluggable `CompletionAdapter` Protocol +
  `OpenAIPassthroughAdapter` + `AnthropicCompletionAdapter`. Multi-provider
  ready.
- **Arch-6** (rc1): `parse_int` / `parse_uuid` canonical names.
  Underscored aliases keep working through v0.8.x via PEP 562
  `__getattr__` + `DeprecationWarning`. Removal in v0.9.0.
- **Arch-2** (rc1): `PromptComposer` with section overrides. Default
  rendering byte-identical to legacy `render_system_prompt`.
- **Arch-1** (rc2): `CandidateContext[ID_T, ItemT]` runtime-checkable
  Protocol. Non-booking consumers can supply their own context dataclass.
- **Arch-3** (this): drop Django runtime dep → `[django]` extra.
- **Arch-4** (this): remove Master\* aliases.

Source: plan-eng-review (Software Architect skill, 2026-05-13).

## [0.8.0-rc2] — 2026-05-14 (PR-2 of 3-PR rollout)

Second PR of the 3-PR v0.8.0 rollout. **Soft-breaking** — only the
orchestrator's runtime check on `context_builder` return type
changes from concrete-class `isinstance(SpecialistContext)` to
structural `isinstance(CandidateContext)`. SpecialistContext callers
keep working unchanged because they already satisfy the Protocol.

### Added

- **`CandidateContext[ID_T, ItemT]` Protocol** (Arch-1 / DRF-685).
  Runtime-checkable structural Protocol. Required attributes:
  `candidates: list[ItemT]`, `candidate_ids: frozenset[ID_T]`,
  `summary_text: str`, `tenant_id: str`. Non-booking consumers (FAQ
  skill, support-ticket skill) implement their own frozen dataclass
  with these fields and pass it to `AIConcierge.send_message` via
  `context_builder` — no longer required to subclass or alias
  `SpecialistContext`.
- **`ItemT` TypeVar** re-exported from `ayla_ai_core` for consumers
  parameterising their own `CandidateContext` impls.

### Changed

- `AIConcierge.send_message` now validates the `context_builder` return
  with `isinstance(..., CandidateContext)` instead of
  `isinstance(..., SpecialistContext)`. The TypeError message updated.
  Internal cast to `SpecialistContext[Any]` after the Protocol check
  preserves the bundled `dispatch_tool_call` path (which uses booking-
  shape fields like `by_id` / `candidate_service_ids` that the Protocol
  intentionally omits). Non-booking consumers should inject their own
  `tool_dispatcher` to handle their custom context shape.

### Soft breaking

The error message for an invalid `context_builder` return type changes
from `"context_builder must return SpecialistContext, got ..."` to
`"context_builder must return a CandidateContext (i.e. an object with
.candidates, .candidate_ids, .summary_text, .tenant_id); got ..."`.
Consumer test suites that grep the message text will need an update.

### Backwards compat

`SpecialistContext[ID_T]` keeps working as the booking-domain concrete
impl; instances pass both `isinstance(x, SpecialistContext)` and
`isinstance(x, CandidateContext)`. Master\* aliases still resolve.

## [0.8.0-rc1] — 2026-05-14 (PR-1 of 3-PR rollout)

First **additive-only** PR of the v0.8.0 architecture refactor. Per the
plan-eng-review (Software Architect skill, 2026-05-13), v0.8.0 lands as
three sequential PRs to keep each independently releasable. rc1 contains
the additive pieces (no break); rc2 adds the generic `CandidateContext`
(soft break); the final v0.8.0 drops the Django runtime dep + Master*
aliases (hard break).

### Added

- **`providers/` module — pluggable CompletionAdapter Protocol**
  (Arch-5 / DRF-689).
  - `CompletionAdapter` Protocol — normalises any provider's response into
    OpenAI-shape that `_parse_completion` already handles
  - `OpenAIPassthroughAdapter` — no-op default (preserves v0.7.x behaviour)
  - `AnthropicCompletionAdapter` — converts Anthropic's `content[].type=tool_use`
    blocks into OpenAI's `tool_calls` shape. Re-serialises `input` dict to
    JSON string (OpenAI contract). Merges multiple text blocks with `\n\n`.
    Handles both attribute-style and dict-style content blocks (Anthropic
    SDK exposes both depending on version).
  - `AIConcierge(..., completion_adapter=...)` — new optional kwarg.
    `ChatResponseDTO.provider` now reads from the adapter (`"openai"` /
    `"anthropic"` / etc.) instead of being hardcoded.
- **`composer.py` — pluggable `PromptComposer`** (Arch-2 / DRF-686).
  Fluent builder with `with_section(...)` / `with_examples(...)`. Default
  rendering for FORMULA_TELA / AYLA voices is **byte-identical** to
  `render_system_prompt(...)` — a regression test in `test_composer.py`
  pins this so 20+ existing replay fixtures keep passing. Sections override
  template slots (currently `masters_summary` — v0.9.0 will expand the
  section-aware template engine to arbitrary names).
- **`parse_int` / `parse_uuid` canonical names** (Arch-6 / DRF-690).
  Renamed from `_safe_int` / `_safe_uuid`. The underscored names remain
  reachable from `ayla_ai_core` via PEP 562 module-level `__getattr__`
  with a `DeprecationWarning`. Internal default kwargs reference the new
  names directly to avoid warning on every dispatch. Scheduled removal of
  underscored aliases: **v0.9.0**.

### Changed

- `AIConcierge.__init__` gains `completion_adapter: CompletionAdapter | None = None`
  (defaults to `OpenAIPassthroughAdapter()`).
- `ChatResponseDTO.provider` populated from the adapter, not hardcoded
  `"openai"`.

### Backwards compat

PR-1 is **purely additive**. v0.7.4 consumers see no signature change at
any public entry point. New `[ai-core]` extras / kwargs default to legacy
behaviour. The migration path for downstream consumers (ai-bot-platform's
`ayla_adapter.py`, `tests/smoke/test_ayla_import.py`) is zero call-site
changes — the pin-bump alone delivers the value.

The next PR (rc2) will introduce `CandidateContext[ID_T, ItemT]` as a
generic of `SpecialistContext` and change one `isinstance` check to a
Protocol; consumers using direct `SpecialistContext` see no break.

## [0.7.4] — 2026-05-13

Post-release hotfix surfaced by an in-depth code review of v0.7.0 → v0.7.3
diff (Code-Reviewer skill, 2026-05-13). Three fixes; none breaking.

### Fixed

- **Encoder cache is now keyed per-model** (Code-Reviewer P0). Pre-v0.7.4
  the module-global `_tiktoken_encoder` pinned to the FIRST model seen by
  the process and silently fed the wrong token counts to every subsequent
  `AIConcierge` using a different model — wrong history truncation,
  wrong dashboard numbers. Replaced with `functools.lru_cache(maxsize=8)`
  keyed on `model_name`. Single-model deployments are unaffected.
- **Token budget now accounts for per-message envelope overhead.** OpenAI
  chat-completions bills ~4 tokens per message + ~3 tokens for the system
  primer. v0.7.3's budget walk counted only `content` and undershot the
  budget by ~10% on a typical 10-message history. Constants
  `_MSG_ENVELOPE_TOKENS = 4` and `_PRIMER_OVERHEAD_TOKENS = 3` now
  participate in the walk. Real prompts now fit the budget.

### Added

- **Re-exported v0.7.3 observability surface from package root.** v0.7.3
  shipped `scope_tenant_id`, `current_tenant_id`, `scope_frozen_now`,
  `current_frozen_now`, `TenantContextFilter`, `ReplayDeterminismError`,
  `set_tenant_id`, `reset_tenant_id`, and `DEFAULT_HISTORY_TOKEN_BUDGET`
  but only from the submodule. Consumers now `from ayla_ai_core import
  scope_tenant_id` (instead of `.observability.scope_tenant_id`) per the
  v0.7.3 release-note implication. Tests pinned in
  `TestObservabilityReExports` so accidental removal fails CI.

### Test plan

Suite: **184/184 pass** (was 180; +4 new regression tests:
encoder-per-model, budget-envelope-respect, observability-re-exports,
default-budget-from-package-root). ruff + mypy clean.

### Migration

None. Pure additive — consumers see no signature change. The encoder
cache and budget changes affect runtime behaviour (more accurate),
not API.

## [0.7.3] — 2026-05-13

Observability + replay-determinism patch. Library logs gain a stable
`tenant_id` field, `ChatResponseDTO` carries telemetry, replay harnesses
get a frozen-clock contract, and per-turn history is now token-bounded.
All four additions are non-breaking for v0.7.2 consumers.

### Added

- **New module** `ayla_ai_core.observability` (DRF-681 / Obs-1)
  - `current_tenant_id()` / `scope_tenant_id()` / `set_tenant_id()` /
    `reset_tenant_id()` — `ContextVar`-backed tenant scope (async-safe).
  - `TenantContextFilter` — auto-populates `record.tenant_id` when a
    third-party log site forgets `extra=`.
  - Every library log record now carries `tenant_id` (5 call sites in
    `orchestrator` + `tool_handlers`). No format-string changes — grep
    of existing log lines still works.

- **Telemetry on `ChatResponseDTO`** (DRF-682 / Obs-2)
  - New optional fields with neutral zero/empty defaults:
    `latency_ms`, `tokens_in`, `tokens_out`, `model`, `provider`.
  - Populated from existing `time.monotonic()` + `completion.usage`
    measurements in `send_message`. No logging change — values are now
    *also* returned, so consumers can build Prometheus / StatsD
    dashboards from the data path instead of grepping log strings.

- **Frozen-clock contract** (DRF-683 / Obs-3)
  - `AIConcierge.send_message(..., frozen_now: datetime | None = None)`
    binds the value via `scope_frozen_now()` for the full LLM round
    trip + tool dispatch. Renderers and history filters can read it via
    `current_frozen_now()` for byte-identical replay.
  - `ReplayDeterminismError(RuntimeError)` — sentinel that consumers
    and custom dispatchers raise when an operation would break replay.

- **History token-budget guard** (DRF-684 / Obs-4)
  - `AIConcierge.__init__(..., history_token_budget: int | None = 4000)`
    caps per-turn history cost. `_compose_messages` walks newest-first,
    keeps messages until the budget is exhausted, drops older ones
    (chronological order preserved in the output to OpenAI).
  - New `[tiktoken]` optional extra for accurate token counting
    (`uv pip install ayla-ai-core[tiktoken]`). When the extra is not
    installed, falls back to a 4-chars-per-token heuristic and emits
    a single `WARNING` log on first use.
  - `token_budget=None` disables the guard entirely (v0.7.2 behaviour).

### Changed

- `_parse_completion` parallel-tool-calls log + `_fallback_clarification`
  warning + `_check_resolver_tenant` opt-out warning + `send_message`
  turn-summary log all now pass `extra={"tenant_id": ...}` (DRF-681).

### Not breaking

All four changes are pure additions. v0.7.2 consumers see no signature
change at any public entry point. `ChatResponseDTO` positional-arg
constructors still compile because new fields are appended with
defaults. The new `[tiktoken]` extra is opt-in.

## [0.7.2] — 2026-05-13

Performance + security hardening from the 6-agent v0.6.0 review. v0.7.1
is reserved as the rapid-response slot for bugs surfaced by ai-bot-platform
Sprint 7 adoption; if it stays empty, that's a good outcome — skip
straight to v0.7.2 in any downstream pin.

### BREAKING (soft)

- **`tool_handlers.handle_confirm_booking` resolvers MUST include `tenant_id`
  in their returned dict** (DRF-680 / Sec-1). v0.7.0 was permissive: a
  resolver that returned `{"name": "Anna"}` silently bypassed the cross-
  tenant guard. v0.7.2 returns `ASK_CLARIFICATION` with reason
  `{master|service}_resolver_no_tenant_id` when `tenant_id` is missing,
  and `{master|service}_tenant_mismatch` (unchanged) when it differs.
  **Migration**: add `tenant_id` to your resolver's row dict. If you
  genuinely can't (legacy wrapper), set
  `my_resolver.__resolver_skips_tenant_check__ = True` — a `WARNING`
  is emitted per call so the bypass stays visible in audit logs.

### Performance

- **O(1) handler lookups** (DRF-677 / Perf-1): `SpecialistContext` gains
  `by_id: dict[ID_T, SpecialistCandidate[ID_T]]` and each
  `SpecialistCandidate` gains `service_id_set: frozenset[ID_T]`. Both
  pre-computed by `build_specialist_context_from_candidates`. Eliminates
  per-call linear scans + set-rebuild that breached the per-turn budget
  at N=1000 (SaaS catalog scale). Backward-compat: `__post_init__`
  auto-populates these fields if a caller constructs `SpecialistContext`
  / `SpecialistCandidate` directly and omits them — no signature break.
- **Parallel `tool_calls` support** (DRF-678 / Perf-2): `orchestrator`
  now dispatches **every** entry in `completion.choices[0].message.tool_calls`
  instead of silently dropping calls beyond `[0]`. gpt-4o emits 2-3
  parallel calls by default; v0.7.0 was discarding production data.
  Merge strategy: "first non-clarification wins" for the primary action
  in `ChatResponseDTO`; remaining results return in a new optional
  `extra_actions: list[{"action_type": str, "action_data": dict}] | None`
  field (`None` for 0/1 tool_calls — fully backward-compatible).

### Security

- **Stricter cross-tenant guard** (DRF-680) — see BREAKING above.

### Defensive

- **`TOOL_DEFINITIONS` is now read-only** (DRF-679 / Perf-3): wrapped via
  `types.MappingProxyType` + outer `tuple`. Mutation attempts
  (`TOOL_DEFINITIONS[0]["x"] = "y"` or `.append(...)`) now raise at the
  top level. `SHOW_MASTERS` / `SHOW_SLOTS` / `CONFIRM_BOOKING` /
  `SHOW_MY_BOOKINGS` / `ASK_CLARIFICATION` aliases inherit the wrap
  (they alias into the tuple). The factory `build_tool_definitions(...)`
  is unchanged — still returns a fresh mutable `list[dict]` per call;
  callers needing mutation switch to it. Note: shallow immutability only;
  nested dicts (`tool["function"]["parameters"]`) remain mutable — left
  for a future minor.

### Internal

- `_parse_completion` return tuple grows by one slot (extra_actions).
  Private API, only `AIConcierge.send_message` consumes it.

### Migration cookbook

```python
# Before (v0.7.0):
def my_master_resolver(master_id, *, tenant_id):
    row = Master.objects.filter(id=master_id, tenant_id=tenant_id).first()
    if row is None:
        return None
    return {"name": row.name, "price_from": row.price_from}

# After (v0.7.2 — add tenant_id):
def my_master_resolver(master_id, *, tenant_id):
    row = Master.objects.filter(id=master_id, tenant_id=tenant_id).first()
    if row is None:
        return None
    return {
        "name": row.name,
        "price_from": row.price_from,
        "tenant_id": row.tenant_id,   # <-- v0.7.2 required (or opt-out attribute)
    }
```

## [0.7.0] — 2026-05-13

### BREAKING

- **`SpecialistContext.tenant_id` is now mandatory** (was `Optional[str] = None`).
  All consumers must pass a non-empty `tenant_id` when constructing a
  `SpecialistContext` directly or via the builder helpers. Empty/missing
  raises `ValueError` at construction time, and `dispatch_tool_call` / every
  `handle_*` function raises `ValueError("tenant_id required …")` if called
  with a context that has empty `tenant_id`. (B3)
- **`build_specialist_context_from_candidates`** and
  **`build_master_context_from_candidates`** now require `tenant_id` as a
  kwarg. The deprecated `build_master_context_from_candidates` is preserved
  (still kwarg-required for migration). (B3)
- **Resolver signatures changed** — `MasterResolver` and `ServiceResolver`
  are now `Callable[..., dict[str, Any] | None]` with a kwarg-only
  `tenant_id: str` parameter the dispatcher passes through. Consumer
  implementations must accept this kwarg (use `def resolver(value, **kwargs)`
  for forward-compat). (B3)
- **`render_system_prompt(..., escape_for_format=True)` default ON.**
  Doubles `{` / `}` in `client_name`, `extra_hint`, `business_name`,
  `business_address`, `off_topic_redirect` before `.format()`. Output for
  inputs containing literal braces will differ from v0.6.x (intentional —
  prevents `KeyError` and template-injection). Callers that have already
  pre-escaped braces in a wrapper layer (e.g. `ai-bot-platform`'s
  `ayla_adapter` DRF-616) pass `escape_for_format=False` to avoid
  double-escape. (B4)

### Security

- **Anti-injection layer in `render_system_prompt`** — new helper
  `_escape_braces` doubles `{`/`}` so `str.format()` reads user-controlled
  fields as literal output, preventing `KeyError` DoS and template-injection
  on stray placeholders in `client_name` / `extra_hint` / brand-config
  strings. (B4)
- **Cross-tenant guard in `handle_confirm_booking`** — when
  `master_resolver` or `service_resolver` returns a dict containing a
  `tenant_id` key, the value is compared to `context.tenant_id`; mismatch
  triggers `ASK_CLARIFICATION` fallback. Defends against stale prompts
  referencing rows from other tenants that slip past the `candidate_ids`
  check. (B3)

### Fixed

- **`_compose_messages` no longer poisons OpenAI history with empty
  assistant turns.** Assistant turns with empty content AND no `tool_calls`
  are now filtered out — OpenAI rejected such messages with HTTP 400 after
  any tool_call appeared in conversation history. Assistant turns WITH
  `tool_calls` are preserved (those carry the function-call payload
  OpenAI needs). (B1)
- **`handle_show_masters` now aligns `match_scores` / `match_reasons` with
  the LLM-emitted index, not the post-filter index.** Previously, when the
  LLM emitted a mix of valid + hallucinated `master_ids`, surviving masters
  inherited scores/reasons that the LLM had assigned to the hallucinated
  IDs. The fix preserves the LLM's endorsement metadata for each surviving
  candidate. Also adds deduplication of repeated IDs (`master_ids=[1, 1, 1]`
  renders one card). **Restores the anti-hallucination layer guarantee** for
  partial-success cases. (B2)

### Changed

- **Tightened Django floor** from `>=5.0,<6.0` to `>=5.2,<6.0`. Django 5.0
  has CVE-2024-39329 (user-enum timing). Consumers on Django 5.0/5.1 must
  upgrade.

### Migration guide for consumers

```python
# v0.6.x — these patterns now break in v0.7.0

# 1. SpecialistContext direct construction:
ctx = SpecialistContext(
    candidates=[...],
    candidate_ids=frozenset(...),
    candidate_service_ids=frozenset(...),
    summary_text="...",
    # tenant_id defaulted to None — broken in v0.7.0
)

# 2. Builder calls:
ctx = build_specialist_context_from_candidates(candidates)
ctx = build_master_context_from_candidates(candidates)

# 3. Resolver implementations:
def my_resolver(value):
    return Master.objects.filter(id=value).first()
```

```python
# v0.7.0 — required call shapes

# 1. SpecialistContext: tenant_id mandatory
ctx = SpecialistContext(
    candidates=[...],
    candidate_ids=frozenset(...),
    candidate_service_ids=frozenset(...),
    summary_text="...",
    tenant_id="formula-tela",  # required, non-empty
)

# 2. Builder calls: tenant_id kwarg required
ctx = build_specialist_context_from_candidates(
    candidates,
    tenant_id="formula-tela",
)

# 3. Resolvers: kwarg-only tenant_id
def my_resolver(value, *, tenant_id: str) -> dict | None:
    return Master.objects.filter(id=value, tenant_id=tenant_id).first()

# Optional: opt out of B4 escape (caller pre-escapes)
prompt = render_system_prompt(
    ...,
    escape_for_format=False,  # caller did doubling already
)
```

### Test budget

- v0.6.0: 123 tests passing
- v0.7.0: **139 tests passing** (+5 B4 TestBraceEscape, +4 B3 TestTenantScoping,
  +4 B1 TestComposeMessages, +3 B2 TestShowMasters alignment)

### Reference

- B1: `src/ayla_ai_core/orchestrator.py::_compose_messages` (lines ~177-189)
- B2: `src/ayla_ai_core/tool_handlers.py::handle_show_masters` (lines ~162-202)
- B3: `src/ayla_ai_core/context.py::SpecialistContext` + `_assert_tenant_id_set`
  in `tool_handlers.py`
- B4: `src/ayla_ai_core/prompts.py::render_system_prompt` +
  `_escape_braces` helper

---

## [0.6.0] — 2026 Q2 (pre-CHANGELOG era)

- `tool_dispatcher` DI hook in `AIConcierge.__init__` (DRF-241) — allows
  consumers to inject their own tool dispatcher (Ayla `show_specialists`
  vs shared `show_masters`) without forking the orchestrator.

## [0.5.x and earlier]

History prior to v0.6.0 lives in git log only. See:
- `feat: BrandVoiceConfig + render_system_prompt (DRF-239)`
- `feat: SpecialistContext generic + multi-tenant + UUID support (DRF-238)`
- `feat: extract AIConcierge + tools + handlers (DRF-237)`
