# Deployment Runbook (Version IDs + Provenance Gate)

Goal: use Version IDs as the canonical proof that the live site is built from this repo snapshot. Asset hash parity is the fallback when you cannot redeploy immediately.

## 0) Azure surfaces (evidence-backed)

Source of record (declared in the UI docs):
- Static hosting: "Azure Static Web Apps (kai-platform-react, www.kelvinale.com)"
- API/runtime: "Azure Container Apps (kai-platform-backend, kai-platform-worker, kai-llm)"
- Registry: "Azure Container Registry (cafbdd83c623acr)"

Evidence: `repo/apps/web/src/pages/ArchitectureInfo.jsx`

App Service: UNKNOWN (no evidence in repo).

## 1) Version ID sources (what the gate verifies)

Backend `/api/version` fields (provenance source):
- `git_sha`: reads `GIT_SHA` or `BUILD_SHA`
- `build_time`: reads `BUILD_TIME` or `BUILD_TIMESTAMP`

Evidence: `repo/services/api/main.py`

Frontend build stamp (provenance source):
- `window.__BUILD__` uses `VITE_BUILD_SHA`, `VITE_BUILD_TIME`, `VITE_APP_VERSION`

Evidence: `repo/apps/web/src/main.jsx`

## 2) Deterministic deployment steps

Step 1: choose build values
- `BUILD_SHA`: commit SHA of this repo snapshot
- `BUILD_TIME`: UTC timestamp (ISO 8601)
- `APP_VERSION`: optional human version string

Example:
- `BUILD_SHA=abcdef1234567890`
- `BUILD_TIME=2026-01-05T12:00:00Z`
- `APP_VERSION=2.0.0`

Step 2: backend deploy (Azure Container Apps)
- Target: "kai-platform-backend" (Container Apps)
- Set env vars (at least one from each pair):
  - `GIT_SHA` or `BUILD_SHA`
  - `BUILD_TIME` or `BUILD_TIMESTAMP`
- Deploy the container image (from ACR or your pipeline).

Step 3: frontend deploy (Azure Static Web Apps)
- Target: "kai-platform-react" (Static Web Apps)
- Build-time env vars for Vite:
  - `VITE_BUILD_SHA`
  - `VITE_BUILD_TIME`
  - `VITE_APP_VERSION`
- Build and deploy the static assets.

Evidence for Vite build:
- `npm run build` runs Vite build.

Source: `repo/apps/web/package.json`

## 3) Provenance gate (must be green)

These are the gate-required env vars:
- `FRONTEND_URL`
- `BACKEND_URL`
- `KAI_ACCESS_PASSWORD`
- `REQUIRE_VERSION_MATCH=true`
- `EXPECTED_FRONTEND_BUILD_SHA`
- `EXPECTED_BACKEND_BUILD_SHA`

Evidence: `repo/tests/e2e/ui_smoke_gate.js`

Recommended command (PowerShell, from repo root):
```
$env:FRONTEND_URL="https://www.kelvinale.com"
$env:BACKEND_URL="https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io"
$env:KAI_ACCESS_PASSWORD="<<SET>>"
$env:REQUIRE_VERSION_MATCH="true"
$env:EXPECTED_FRONTEND_BUILD_SHA="<<BUILD_SHA>>"
$env:EXPECTED_BACKEND_BUILD_SHA="<<BUILD_SHA>>"
.\scripts\run_build_and_integrity.ps1
```

Evidence for backend URL used by the app:
`repo/apps/web/src/config.js`

Gate output is written to a timestamped folder under:
`repo/integrity_runs/<timestamp>/`

Evidence: `repo/scripts/run_build_and_integrity.ps1`

PASS criterion: the run exits `0` and `results.json` shows all gates pass.

## 4) Safe write tests (optional)

By default, write tests are blocked unless explicitly allowed.

To allow write smoke tests:
- `ALLOW_WRITE_SMOKE=true`
- If backend URL is prod, also set `ALLOW_PROD_WRITE_SMOKE=true`

Evidence: `repo/tests/e2e/integrity_suite.js`

## 5) Asset hash parity (fallback when you cannot redeploy)

If you cannot redeploy immediately, do a parity check:

1) Build locally:
```
cd repo/apps/web
npm run build
```

2) Compare asset filenames in:
- `repo/apps/web/dist/index.html` (local)
- `https://www.kelvinale.com` (live)

If the hashed asset filenames match, parity is likely (not canonical).

Evidence for build output location:
`repo/apps/web/vite.config.js`

