# Owner Readiness Checklist (Kai / Klaudit System)

Audience: **Business Owner** and **Technical Admin**.
Format: **Step‑by‑step**, with clear examples.

---

## Step 0 — What You’re Getting (Quick Summary)
You’re receiving a **complete app** (UI + API + Worker) that can run locally or in Azure.
No legacy data or secrets are included.

Included:
- UI: `apps/web`
- API: `services/api`
- Worker: `services/worker`
- Infra + scripts: `infra/`, `scripts/`
- Verification: `scripts/verification/verify_system_integrity.ps1`

---

# A) Business Owner Steps (Non‑Technical)

## Step 1 — Choose Hosting
- **Local App** (test)
- **Azure App** (production)

**Decision:** Which hosting do you want first? (Local recommended for a quick sanity check.)

## Step 2 — Provide Access Credentials to Technical Admin
You need to give **new owner credentials** (not yours):
- Azure Subscription ID
- Azure Tenant ID
- Resource Group name
- A temporary **Owner** or **Contributor** role during migration

## Step 3 — Decide Which Features to Enable
These features increase capability but require keys:
- Azure OpenAI (optional)
- SA360 (optional)
- Vector Search (optional)

If you don’t have keys yet, the system still runs, but those features stay off.

---

# B) Technical Admin Steps (Technical)

## Step 1 — Configure Environment Variables
Use `.env.example` as your source of truth.
Create `.env` with real values.

**Required (core system):**
- `KAI_BACKEND_URL`
- `KAI_FRONTEND_URL`
- `KAI_ACCESS_PASSWORD`
- `KAI_ENV_GUI_PASSWORD`

**Required for audits + job queue:**
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_STORAGE_ACCOUNT`
- `AZURE_STORAGE_CONTAINER`

### Optional (enable only if you have credentials)
**Azure OpenAI** (adds higher‑quality verbalization)
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`

**Local LLM** (offline reasoning)
- `LOCAL_LLM_ENDPOINT`
- `LOCAL_LLM_MODEL`

**SA360** (account data retrieval)
- `SA360_CLIENT_ID`
- `SA360_CLIENT_SECRET`
- `SA360_REFRESH_TOKEN`
- `SA360_LOGIN_CUSTOMER_ID` (optional, MCC)
- `SA360_OAUTH_REDIRECT_URI`

**Vector Search** (knowledge retrieval)
- `CONCIERGE_SEARCH_ENDPOINT`
- `CONCIERGE_SEARCH_KEY`
- `CONCIERGE_SEARCH_INDEX`

---

## Step 2 — Run Locally (Fast sanity check)
**API:**
- `python services/api/main.py`

**Worker (if using async jobs):**
- `python services/worker/jobs_worker.py`

**UI:**
- `cd apps/web` → `npm install` → `npm run dev`

Verify:
- `/api/health`
- `/api/diagnostics/health`
- `/api/settings/env`

---

## Step 3 — Run Docker (App‑first bundle)
See `docker/README.md` and `docker/docker-compose.yml`.

---

## Step 4 — Deploy to Azure
- Backend API → Azure Container App
- Worker → Azure Container App
- Frontend → Static Web App / Storage static site

Set env vars from `.env.example`.

---

## Step 5 — Verify
Run:
- `scripts/verification/verify_system_integrity.ps1 -Mode azure`

Confirm:
- `verification_runs/<timestamp>/results.json` shows all required gates passing.

---

# C) What “Working” Looks Like
A new owner should be able to:
- Start API + UI + Worker
- Call `/api/health` and `/api/diagnostics/health`
- Generate an audit via `/api/audit/generate` or `/api/audit/upload`
- Use `/api/settings/env` to verify the Env GUI works

---

# D) Ownership Transfer Checklist (Summary)
- ✅ Azure subscription + resource group ready
- ✅ Storage account + containers ready
- ✅ Required env vars set
- ✅ Optional keys set (if needed)
- ✅ Verification run is green
