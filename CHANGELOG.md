# Changelog

All notable changes to `ayla-ai-core` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with **pre-1.0 caveats** — minor releases may be breaking. Each release ships a
migration guide. Consumers pin by SHA (not tag — tags are force-pushable).

---

## [Unreleased]

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
