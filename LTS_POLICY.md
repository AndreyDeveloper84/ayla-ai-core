# `ayla-ai-core` LTS Policy

Companion to [`RELEASING.md`](./RELEASING.md). Defines the support
commitment for v1.x and later major lines.

## Promise

> Every `1.x` release line is supported with **bug fixes and security
> patches for a minimum of 12 months** from its initial release.
> Breaking changes only ship in the next major (`2.0.0`).

## What "supported" means

For any version line `X.y.z` while in support:

| Type | Cadence | Where |
|---|---|---|
| **Critical security patches** | Within 14 days of internal triage | New `X.y.z+1` release on the support line |
| **High-severity bug fixes** | Within 30 days | Same as above |
| **Public API surface changes** | NEVER inside a major line — only `X+1.0.0` | — |
| **Behaviour changes that consumers can observe** | Only when they preserve API contract (e.g., perf improvements that don't change return values, cache invalidation that doesn't change outputs) | New minor `X.y+1.0` |

## What "supported" does NOT mean

- We do NOT backport **new features** to older major lines. v1.5.0
  features stay in v1.x; they do NOT appear in v0.x or v2.x.
- We do NOT promise behavioural fidelity across **major** boundaries.
  v2.0.0 may reorganise internals (the planned plugin architecture,
  for instance). Consumers migrate using `ayla-migrate` + CHANGELOG.
- We do NOT promise that **deprecated** symbols keep working forever.
  Per [`RELEASING.md`](./RELEASING.md) the deprecation window is one
  minor; removal lands in the next major.

## Version line support table

| Major | Initial release | End-of-support (minimum) | Status |
|---|---|---|---|
| `0.x` (pre-1.0) | 2026-04 | **No support** — pre-1.0 had explicit "minor may break" caveat | Deprecated |
| `1.x` | 2026-05 (target) | 2027-05 (minimum) | **Active** |
| `2.x` | TBD | 12+ months after `2.0.0` | Future |

## Security disclosure

Critical vulnerabilities (cross-tenant data leak, prompt injection
bypass, RCE) follow coordinated disclosure:

1. **Report**: email `security@drfproject.example` (or open a GitHub
   Security Advisory if the issue is non-exploitable in isolation)
2. **Triage**: we acknowledge within 72 hours
3. **Fix**: developed privately on a security advisory branch
4. **Coordinated release**: patch SHA shipped + advisory published +
   consumers notified
5. **Public CHANGELOG**: entry added after consumers have updated

## Out-of-support behaviour

When a version line moves to **End-of-support**:

- The line keeps existing on GitHub (tags + branches remain pinnable)
- No new patches — including security
- A `SECURITY.md` notice points to the supported lines
- Consumers must upgrade to a supported line

## Versioning rules recap (post-1.0)

| Bump | Triggered by |
|---|---|
| `1.X.Y` → `1.X.Y+1` (PATCH) | Bug fixes, security patches, internal refactors |
| `1.X.Y` → `1.X+1.0` (MINOR) | New additive features (new `__all__` symbols, new optional kwargs with safe defaults) |
| `1.X.Y` → `2.0.0` (MAJOR) | Removed symbols from `__all__`, signature changes, breaking behaviour |

The public API surface is pinned by
`tests/test_public_api_surface.py::V1_PUBLIC_API`. CI fails loud on
unintentional adds/removes.
