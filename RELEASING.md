# Releasing `ayla-ai-core`

Public release process and policies for the library. Companion document:
[`LTS_POLICY.md`](./LTS_POLICY.md).

## Versioning

`ayla-ai-core` follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

| Component | What changes it | Examples (post-1.0) |
|---|---|---|
| **MAJOR** | Breaking API changes — removed symbols from `__all__`, signature changes, behavioural changes that break existing consumers | `2.0.0`: removed `MasterContext` (already done in v0.8.0 pre-1.0) |
| **MINOR** | New features added to `__all__`; no breaks | `1.1.0`: new `OpenRouterCompletionAdapter` provider |
| **PATCH** | Bug fixes, security patches, internal refactors, docstring fixes | `1.0.1`: encoder cache fix |

### Pre-1.0 caveats (historical)

Versions `0.x.y` did NOT follow strict semver — minor bumps could break.
This is intentional: pre-1.0 explicitly signalled "API not yet stable."
The v1.0.0 release **freezes** the public API (see
[`LTS_POLICY.md`](./LTS_POLICY.md) for the commitment) and from that
point forward the rules above apply strictly.

## Release process

### 1. Pre-release verification

Run on the release branch (e.g. `release/vX.Y.Z`):

```bash
uv run pytest -v             # full suite must pass
uv run mypy src/             # 0 errors
uv run ruff check src/ tests/  # all clean
```

For minor/major bumps, also verify:

- [ ] `CHANGELOG.md` has a complete section under `[X.Y.Z] — YYYY-MM-DD`
- [ ] `pyproject.toml` `version` is bumped
- [ ] `src/ayla_ai_core/__init__.py` `__version__` is bumped
- [ ] `tests/test_public_api_surface.py::V1_PUBLIC_API` snapshot updated
      if `__all__` changed (CI fails loud otherwise)
- [ ] Migration steps documented in CHANGELOG for ANY breaking change

### 2. Open PR + merge

```bash
gh pr create --base main --head release/vX.Y.Z \
  --title "release: vX.Y.Z — <one-line summary>"
# Wait for CI green
gh pr merge --squash --delete-branch
```

### 3. Tag

```bash
git checkout main && git pull origin main
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z

# Capture the commit SHA (NOT the annotated-tag object SHA) for
# downstream consumers' pin-bumps:
git rev-parse vX.Y.Z^{commit}
```

### 4. Downstream pin-bump

For every known consumer (`ai-bot-platform`, future Ayla marketplace
SaaS, etc.), open a `chore(deps)` PR that:

1. Replaces the SHA in `pyproject.toml` `[ai-core]` extra
2. Runs `uv lock --upgrade-package ayla-ai-core`
3. Updates the version assertion in `tests/smoke/test_ayla_import.py`

**Pin format**: always 40-char commit SHA, never the tag. Tags are
force-pushable on GitHub; SHAs are immutable (Security P1 from the
2026-05 six-agent audit).

## Security patches

Critical-severity vulnerabilities (RCE, prompt injection bypass,
cross-tenant data leak) are released as **patch versions** on every
supported `x.y` line per [`LTS_POLICY.md`](./LTS_POLICY.md).

Process:
1. Private fix in a security advisory branch
2. Coordinated disclosure with downstream consumers
3. Patch release + GitHub Security Advisory
4. Public CHANGELOG entry only AFTER all downstream consumers have
   pinned the patch SHA

## Deprecation windows

A symbol marked deprecated in version `X.Y.Z` is **removed no earlier
than `X+1.0.0`** (the next major). Between deprecation and removal:

- The symbol keeps working
- Any access (import or call) emits a `DeprecationWarning` with
  `stacklevel=2` so consumer code shows in the warning, not the library
- The CHANGELOG entry for the deprecating version names the
  replacement explicitly
- The `ayla-migrate` CLI tool ships a rewrite rule for the symbol so
  consumers can update mechanically (`ayla-migrate --from X.Y --to N.0`)

## CI gates

Every PR to `main` must pass:

- `pytest -v` on Python 3.12 and 3.13
- `mypy src/` (0 errors, strict-ish per `pyproject.toml`)
- `ruff check src/ tests/` (clean)
- `tests/test_public_api_surface.py` (frozen `__all__` snapshot)
- Build the package wheel without errors

The public-API snapshot test is the **load-bearing gate** — it catches
accidental signature drift / missing exports before they reach
consumers.
