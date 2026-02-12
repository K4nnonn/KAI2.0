# Kai Transfer Package (Handoff)

This repo is a clean, transferrable snapshot of the Kai system intended for a new owner.
It is **app-first** and **domain-independent**. It can run locally or be deployed to your Azure.

This document is aligned to the current codebase and verification scripts in this package.

## What This Package Contains
- **Frontend UI**: `apps/web` (Vite UI)
- **Backend API**: `services/api` (FastAPI)
- **Job worker**: `services/worker` (queue/table/worker loop)
- **Infra + deployment helpers**: `infra/`, `docker/`, `scripts/`
- **Handoff artifacts**: `handoff/` (checklist + PDF)
- **Verification**: `scripts/verification/verify_system_integrity.ps1`

## Entry Points (Evidence)
These endpoints exist in `services/api/main.py`:
- Health: `/api/health`, `/api/diagnostics/health`
- Chat: `/api/chat/route`, `/api/chat/send`, `/api/chat/history`
- Audit: `/api/audit/generate`, `/api/audit/upload`, `/api/audit/download/{filename}`
- SA360: `/api/sa360/accounts`, `/api/integrations/sa360/fetch`, `/api/integrations/sa360/fetch-and-audit`
- Diagnostics: `/api/diagnostics/advisor`, `/api/diagnostics/tone`, `/api/diagnostics/verbalizer`
- Settings: `/api/settings`, `/api/settings/env`, `/api/settings/env/update`

## Minimum Requirements (Evidence)
Required inputs are visible in `.env.example` and in code (see `services/api` and `services/worker`):

### Required for basic API + UI
- `KAI_BACKEND_URL`
- `KAI_FRONTEND_URL`
- `KAI_ACCESS_PASSWORD`
- `KAI_ENV_GUI_PASSWORD`

### Required for storage-based audits / job queue (worker)
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_STORAGE_ACCOUNT`
- `AZURE_STORAGE_CONTAINER`

Worker queue/table details are configurable via envs in:
- `services/worker/utils/job_queue.py`
- `services/worker/jobs_worker.py`

Key envs include:
- `JOB_QUEUE_NAME`, `JOB_TABLE_NAME`, `JOB_RESULT_CONTAINER`
- `JOB_MAX_RUNTIME_SECONDS`, `JOB_QUEUE_VISIBILITY_TIMEOUT`, `JOB_QUEUE_POLL_SECONDS`

### Optional (only if you enable that capability)
- Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- Local LLM: `LOCAL_LLM_ENDPOINT`, `LOCAL_LLM_MODEL`
- SA360 (OAuth-only): `SA360_CLIENT_ID`, `SA360_CLIENT_SECRET`, `SA360_REFRESH_TOKEN`, `SA360_LOGIN_CUSTOMER_ID`, `SA360_OAUTH_REDIRECT_URI`
- Vector Search (if used): `CONCIERGE_SEARCH_ENDPOINT`, `CONCIERGE_SEARCH_KEY`, `CONCIERGE_SEARCH_INDEX`
- License hardening: `LICENSE_SERVER_URL`, `LICENSE_INSTANCE_ID`, `LICENSE_ENFORCEMENT_MODE`, `LICENSE_REFRESH_INTERVAL_HOURS`, `LICENSE_RENEW_DAYS_BEFORE_EXP`, `LICENSE_GRACE_DAYS`, `LICENSE_REQUEST_TIMEOUT_SECONDS`, `LICENSE_CACHE_PATH`, `BRAIN_GATE_USE_OBFUSCATED`

## License Enforcement Modes
- `off`: no request blocking; behaves as current open mode.
- `soft`: requests continue, but warning header may be attached (`X-License-Warn`).
- `hard`: protected routes return HTTP `503` with `{"status":"license_blocked","reason":"..."}` when license is invalid and outside grace.

Exempt routes in all modes:
- `/api/health`
- `/api/diagnostics/health`
- `/api/auth/verify`
- `/docs`
- `/openapi.json`

Diagnostics:
- `/api/diagnostics/license` returns mode, validity, reason, expiration, cache status, and refresh timing.

## Optional Obfuscation Build (brain_gate only)
Build the licensing module to a compiled artifact:
- `powershell -ExecutionPolicy Bypass -File .\\scripts\\build_brain_gate_obfuscated.ps1`

Runtime toggle:
- `BRAIN_GATE_USE_OBFUSCATED=true` requires a compiled `brain_gate*.pyd`/`brain_gate*.so` in `build/obf`.

## Local Run (App-First)
1) Copy `.env.example` to `.env` and fill values for your environment.
2) Start API:
   - `python services/api/main.py`
3) Start worker (if using async jobs):
   - `python services/worker/jobs_worker.py`
4) Start UI:
   - `cd apps/web` then `npm install` and `npm run dev`

## Docker Run (App-First)
- See `docker/README.md` and `docker/docker-compose.yml` (wrapper) or root `docker-compose.yml`.
- `infra/docker/Dockerfile.web` builds the UI and serves it via Nginx.

## Local LLM (Ollama + Qwen) Setup
Docker compose includes an Ollama service and will pull the configured model on first start.

Docker:
1) Set in `.env`:
   - `LOCAL_LLM_ENDPOINT=http://ollama:11434`
   - `LOCAL_LLM_MODEL=qwen2.5:14b`
2) `docker compose up --build`

Local install (no Docker):
1) Install Ollama from https://ollama.com
2) Pull a model (example): `ollama pull qwen2.5:14b` (or your preferred Qwen model)
3) Set:
   - `LOCAL_LLM_ENDPOINT=http://localhost:11434`
   - `LOCAL_LLM_MODEL=qwen2.5:14b`

## Azure Deploy (App-First)
- Backend API and Worker are container apps.
- UI is static web hosting.
- Set env vars from `.env.example` (and optional envs above).

## Verification (Evidence)
Run:
- `scripts/verification/verify_system_integrity.ps1 -Mode azure`
This writes a run folder under `verification_runs/` and includes `results.json`.

## What’s Included
- Full source code (`apps/`, `services/`, `scripts/`, `infra/`, `tests/`)
- Handoff checklist (`handoff/owner_readiness_checklist.md` + `.html` + `.pdf`)
- Manifest (`MANIFEST_SHA256.txt`) and secret scan (`SECRET_SCAN_REPORT.txt`)
