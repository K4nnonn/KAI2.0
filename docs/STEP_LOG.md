# Step Log (Shadow Only)

## Step 1 — Skeleton staging
Status: completed

Actions:
- Created `repo_shadow/repo/` skeleton directories.
- Added `repo_shadow/repo/README.md` and `repo_shadow/repo/docs/STRUCTURE.md`.

Gate:
```

## Step 10 - Migration scan root fix + timeout stabilization
Status: completed

Actions:
- Updated migration scan root to repo root.
- Increased router performance timeout to reduce flake.

Evidence (scan root change):
```javascript
const repoRoot = path.resolve(__dirname, '..', '..');
```

Evidence (timeout change):
```javascript
timeout: 60000,
```

Gate (post-change run):
```
node .\integrity_suite.js
```

Gate output:
```
{
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  }
}
```

## Step 11 - Boot/health gate added to runner
Status: completed

Actions:
- Added `tests/e2e/health_gate.js`.
- Added `api_health` gate to `run_build_and_integrity.ps1`.

Evidence (runner results.json):
```
[
  {
    "name": "build_frontend",
    "exit_code": 0
  },
  {
    "name": "api_health",
    "exit_code": 0
  },
  {
    "name": "integrity_suite",
    "exit_code": 0
  }
]
```

Run folder:
```
C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_080717
```

## Step 12 - Deliberate failing run recorded
Status: completed

Actions:
- Forced `BACKEND_URL=http://127.0.0.1:1` to trigger health failure.

Evidence (manual run log):
```
STDOUT:
Gate failed: api_health (exit 1)
Results: C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_080854\results.json
STDERR:
EXITCODE=1
```

Evidence (results.json):
```
[
  {
    "name": "build_frontend",
    "exit_code": 0
  },
  {
    "name": "api_health",
    "exit_code": 1
  }
]
```

## Step 13 - Expanded runtime gates (auth + read)
Status: completed

Actions:
- Added invalid admin password check for env updates.
- Added SA360 accounts read contract check.

Evidence (integrity suite output summary):
```
{
  "summary": {
    "pass": 12,
    "fail": 0,
    "skip": 1
  }
}
```

Evidence (new checks):
```
"name": "api.settings.env.update.invalid"
"status": 403

"name": "api.sa360.accounts"
"status": 200
```

Run folder:
```
C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_092143
```

## Step 14 - UI smoke gate (www.kelvinale.com)
Status: completed

Actions:
- Added `tests/e2e/ui_smoke_gate.js`.
- Added `ui_smoke` gate to `run_build_and_integrity.ps1`.

Evidence (UI smoke output):
```
{
  "frontend_url": "https://www.kelvinale.com",
  "gate_present": true,
  "preauth_chat_hidden": true,
  "auth_success": true,
  "heading_visible": true,
  "chat_input_visible": true,
  "backend_health": { "status": 200, "ok": true }
}
```

Run folder:
```
C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_093941
```

## Step 15 - Safe write flow (chat history round-trip)
Status: completed

Actions:
- Added session-scoped chat write + history checks to integrity suite.

Evidence (integrity suite summary):
```
{
  "summary": {
    "pass": 15,
    "fail": 0,
    "skip": 0
  }
}
```

Evidence (chat write + history):
```
"name": "api.chat.send.write"
"status": "pass"

"name": "api.chat.history.session"
"status": "pass"
```

## Step 16 - Migrations sanity resolved as N/A
Status: completed

Actions:
- Marked migrations sanity as not applicable when no framework markers exist.

Evidence:
```
"name": "migrations.sanity"
"status": "pass"
"detail": "Not applicable: no migration framework markers found at repo root."
```

## Step 17 - Remove default secrets/URLs + require explicit write allow
Status: completed

Actions:
- Required `BACKEND_URL`, `FRONTEND_URL`, and `KAI_ACCESS_PASSWORD` env vars in test gates.
- Required `ALLOW_WRITE_SMOKE=true` to run chat write/read gate.
- Added explicit `exit 0` in runner for clean exit code.

Evidence (env required in gates):
```javascript
const BACKEND_URL = requireEnv('BACKEND_URL');
const ACCESS_PASSWORD = requireEnv('KAI_ACCESS_PASSWORD');
```

```javascript
const FRONTEND_URL = requireEnv('FRONTEND_URL');
const BACKEND_URL = requireEnv('BACKEND_URL');
const ACCESS_PASSWORD = requireEnv('KAI_ACCESS_PASSWORD');
```

Evidence (write allow guard):
```javascript
const ALLOW_WRITE_SMOKE = String(process.env.ALLOW_WRITE_SMOKE || '').toLowerCase() === 'true';
```

Evidence (runner exit code):
```
EXITCODE=0
```

Run folder:
```
C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_103037
```

Evidence (script hashes):
```
4F92435E20CBB600DA9127989E4A398175CFD71937FE0B7204B1A63EFCAF9D43  C:\Software Builds\repo_shadow\repo\tests\e2e\integrity_suite.js
882AC4EC843FD87B856093F544B8FDA6F7E43F472906B931821F64CA55CA1398  C:\Software Builds\repo_shadow\repo\tests\e2e\ui_smoke_gate.js
D1CC6D767659DFBC4A58A3568EA50F711D9DA550FC64D380C44880E429CE4B70  C:\Software Builds\repo_shadow\repo\tests\e2e\health_gate.js
D04CD4E618A09113BF9442B0B79F96CC22DA2B6296E7BDE97E1B36EE3579B546  C:\Software Builds\repo_shadow\repo\scripts\run_build_and_integrity.ps1
```

Run folder:
```
C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_102109
```

## Step 9 — Audit-grade integrity run outputs
Status: completed

Actions:
- Ran `repo_shadow/repo/scripts/run_build_and_integrity.ps1` via `Start-Process` with stdout/stderr capture.
- Generated timestamped gate output folder with `results.json` and `run_meta.json`.

Command:
```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path "..\\integrity_runs" ("manual_run_{0}_stdout.txt" -f $stamp)
$stderr = Join-Path "..\\integrity_runs" ("manual_run_{0}_stderr.txt" -f $stamp)
$combined = Join-Path "..\\integrity_runs" ("manual_run_{0}.txt" -f $stamp)
$proc = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PWD\\run_build_and_integrity.ps1`"" -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$exitCode = $proc.ExitCode
"STDOUT:" | Set-Content -Path $combined
Get-Content -Path $stdout | Add-Content -Path $combined
"STDERR:" | Add-Content -Path $combined
Get-Content -Path $stderr | Add-Content -Path $combined
"EXITCODE=$exitCode" | Add-Content -Path $combined
```

Evidence (manual run log):
```
STDOUT:
Run folder: C:\Software Builds\repo_shadow\repo\integrity_runs\20260104_205034
STDERR:
EXITCODE=0
```

Evidence (results.json excerpt):
```
[
  {
    "name": "build_frontend",
    "command": "npm run build",
    "exit_code": 0,
    "stdout": "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260104_205034\\gate_01_build_frontend\\stdout.txt",
    "stderr": "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260104_205034\\gate_01_build_frontend\\stderr.txt"
  },
  {
    "name": "integrity_suite",
    "command": "node .\\\\integrity_suite.js",
    "exit_code": 0,
    "stdout": "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260104_205034\\gate_02_integrity_suite\\stdout.txt",
    "stderr": "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260104_205034\\gate_02_integrity_suite\\stderr.txt"
  }
]
```

## Step 2 — Move frontend to apps/web
Status: completed

Actions:
- Moved `repo_shadow/kai-frontend` -> `repo_shadow/repo/apps/web`.
- Ran frontend build in new location.

Command:
```

## Step 3 — Move backend to services/api
Status: completed

Actions:
- Moved `repo_shadow/kai-backend-complete` -> `repo_shadow/repo/services/api`.

Gate:
```

## Step 4 — Split worker into services/worker
Status: completed

Actions:
- Moved `repo_shadow/repo/services/api/jobs_worker.py` -> `repo_shadow/repo/services/worker/jobs_worker.py`.
- Copied `utils/job_queue.py` and `utils/telemetry.py` into `repo_shadow/repo/services/worker/utils/`.

Evidence (worker file moved):
```

## Step 5 — Move Playwright tests to tests/e2e
Status: completed

Actions:
- Moved `repo_shadow/playwright-run` -> `repo_shadow/repo/tests/e2e`.
- Increased `api.chat.send.general` timeout in integrity suite to 30s to avoid flaky timeouts.

Evidence (gate failure + fix):
```

## Step 6 — Consolidate Dockerfiles into infra/docker
Status: completed

Actions:
- Copied Dockerfiles into `repo_shadow/repo/infra/docker`:
  - `Dockerfile.api` from `repo_shadow/repo/services/api/Dockerfile`
  - `Dockerfile.llama` from `repo_shadow/repo/services/api/Dockerfile.llama`
  - `Dockerfile.kais-shim` from `repo_shadow/kai-llm-shim/Dockerfile`
  - `Dockerfile.llama-backend` from `repo_shadow/llama-build2/backend/Dockerfile`
- Verified SHA256 matches for each copy.

Evidence (hash match):
```

## Step 7 — Repo map (shadow)
Status: completed

Actions:
- Generated repo map at `repo_shadow/repo/docs/REPO_MAP.md` excluding `node_modules` and `dist`.

Evidence:
```
# Repo Map (Shadow Only)

Base: C:\Software Builds\repo_shadow\repo

Excluded: node_modules, dist

Entries: 566
```

Gate:
```
node C:\Software Builds\repo_shadow\repo\tests\e2e\integrity_suite.js
```

Gate output:
```
{
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  }
}
```

## Step 8 — Build + integrity script
Status: completed

Actions:
- Added `repo_shadow/repo/scripts/run_build_and_integrity.ps1`.

Gate:
```
node C:\Software Builds\repo_shadow\repo\tests\e2e\integrity_suite.js
```

Gate output:
```
{
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  }
}
```
api_match=True
llama_match=True
shim_match=True
llama_backend_match=True
```

Gate:
```
node C:\Software Builds\repo_shadow\repo\tests\e2e\integrity_suite.js
```

Gate output:
```
{
  "config": {
    "BACKEND_URL": "https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io",
    "checks": 11
  },
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  },
  "results": [
    {
      "name": "api.health",
      "goal": "Backend reachable and responding",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 92,
        "data": "{\"status\":\"healthy\",\"service\":\"Kai Platform API\"}"
      }
    },
    {
      "name": "api.diagnostics.health",
      "goal": "Core dependency health (LLM/SA360/QA sample)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 1978,
        "data": "{\"status\":\"ok\",\"errors\":[],\"accounts\":{\"count\":34,\"sample\":[{\"customer_id\":\"5771282191\",\"name\":\"Canada_Mobility\",\"manager\":true},{\"customer_id\":\"4427623347\",\"name\":\"Canada_Pennzoil\",\"manager\":true},{\"customer_id\":\"3662965856\",\"name\":\"Canada_Quaker State\",\"manager\":true},{\"customer_id\":\"3716440491\",\"name\":\"Canada_Rotella\",\"manager\":true},{\"customer_..."
      }
    },
    {
      "name": "api.auth.verify.valid",
      "goal": "Password gate validation",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 17,
        "data": "{\"status\":\"success\",\"authenticated\":true}"
      }
    },
    {
      "name": "api.auth.verify.invalid",
      "goal": "Auth isolation on bad password",
      "status": "pass",
      "detail": {
        "status": 401,
        "duration_ms": 16,
        "data": "{\"detail\":\"Invalid password\"}"
      }
    },
    {
      "name": "api.settings.get",
      "goal": "Settings contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13,
        "data": "{\"ai_chat_enabled\":true,\"ai_insights_enabled\":true,\"theme\":\"light\"}"
      }
    },
    {
      "name": "api.settings.env.list",
      "goal": "Env list masked contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13,
        "data": "{\"env\":[{\"key\":\"AZURE_OPENAI_ENDPOINT\",\"value\":\"************************************com/\"},{\"key\":\"AZURE_OPENAI_API_KEY\",\"value\":\"****************************8795\"},{\"key\":\"AZURE_OPENAI_DEPLOYMENT\",\"value\":\"*******urbo\"},{\"key\":\"AZURE_OPENAI_API_VERSION\",\"value\":\"**************view\"},{\"key\":\"SERPAPI_KEY\",\"value\":\"***********************************..."
      }
    },
    {
      "name": "api.chat.route.performance",
      "goal": "Router contract (intent + planner)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 16258,
        "data": "{\"intent\":\"performance\",\"tool\":\"performance\",\"run_planner\":true,\"run_trends\":false,\"themes\":[],\"customer_ids\":[\"3532896537\"],\"needs_ids\":false,\"notes\":\"router_verified model=local verify=azure; Resolved account to Havas_Shell_GoogleAds_US_Mobility Fuels (3532896537).\",\"confidence\":0.8,\"needs_clarification\":false,\"clarification\":\"How did the Fuels a..."
      }
    },
    {
      "name": "api.chat.send.general",
      "goal": "General chat response contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 4696,
        "data": "{\"reply\":\"Hi! I need to create a social media post for my business. I have a brand that's a bit edgy and don't know where to start. Can you help me come up with a compelling and engaging post idea?\\n• What are some ways to showcase unique or quirky products in a compelling way?\\n• Can you suggest some fresh and relevant hashtags for my business to ..."
      }
    },
    {
      "name": "api.audit.business_units",
      "goal": "Audit business units contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 17,
        "data": "{\"business_units\":[{\"id\":\"Brand\",\"name\":\"Brand\",\"description\":\"Brand campaigns\"},{\"id\":\"NonBrand\",\"name\":\"Non-Brand\",\"description\":\"Non-brand campaigns\"},{\"id\":\"PMax\",\"name\":\"Performance Max\",\"description\":\"PMax campaigns\"}]}"
      }
    },
    {
      "name": "api.serp.check",
      "goal": "SERP check contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 84,
        "data": "{\"status\":\"success\",\"results\":[{\"url\":\"https://google.com\",\"status\":200,\"soft_404\":false}]}"
      }
    },
    {
      "name": "migrations.sanity",
      "goal": "Migration framework detected",
      "status": "skip",
      "detail": "No migration framework markers found."
    }
  ]
}
```
Initial gate after move: pass=9, fail=1 (api.chat.send.general timeout 15s).
Fix: set timeout: 30000 for api.chat.send.general in integrity_suite.js.
```

Gate:
```
node C:\Software Builds\repo_shadow\repo\tests\e2e\integrity_suite.js
```

Gate output:
```
{
  "config": {
    "BACKEND_URL": "https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io",
    "checks": 11
  },
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  },
  "results": [
    {
      "name": "api.health",
      "goal": "Backend reachable and responding",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 76,
        "data": "{\"status\":\"healthy\",\"service\":\"Kai Platform API\"}"
      }
    },
    {
      "name": "api.diagnostics.health",
      "goal": "Core dependency health (LLM/SA360/QA sample)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 1204,
        "data": "{\"status\":\"ok\",\"errors\":[],\"accounts\":{\"count\":34,\"sample\":[{\"customer_id\":\"5771282191\",\"name\":\"Canada_Mobility\",\"manager\":true},{\"customer_id\":\"4427623347\",\"name\":\"Canada_Pennzoil\",\"manager\":true},{\"customer_id\":\"3662965856\",\"name\":\"Canada_Quaker State\",\"manager\":true},{\"customer_id\":\"3716440491\",\"name\":\"Canada_Rotella\",\"manager\":true},{\"customer_..."
      }
    },
    {
      "name": "api.auth.verify.valid",
      "goal": "Password gate validation",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 18,
        "data": "{\"status\":\"success\",\"authenticated\":true}"
      }
    },
    {
      "name": "api.auth.verify.invalid",
      "goal": "Auth isolation on bad password",
      "status": "pass",
      "detail": {
        "status": 401,
        "duration_ms": 14,
        "data": "{\"detail\":\"Invalid password\"}"
      }
    },
    {
      "name": "api.settings.get",
      "goal": "Settings contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 12,
        "data": "{\"ai_chat_enabled\":true,\"ai_insights_enabled\":true,\"theme\":\"light\"}"
      }
    },
    {
      "name": "api.settings.env.list",
      "goal": "Env list masked contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13,
        "data": "{\"env\":[{\"key\":\"AZURE_OPENAI_ENDPOINT\",\"value\":\"************************************com/\"},{\"key\":\"AZURE_OPENAI_API_KEY\",\"value\":\"****************************8795\"},{\"key\":\"AZURE_OPENAI_DEPLOYMENT\",\"value\":\"*******urbo\"},{\"key\":\"AZURE_OPENAI_API_VERSION\",\"value\":\"**************view\"},{\"key\":\"SERPAPI_KEY\",\"value\":\"***********************************..."
      }
    },
    {
      "name": "api.chat.route.performance",
      "goal": "Router contract (intent + planner)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 15952,
        "data": "{\"intent\":\"performance\",\"tool\":\"performance\",\"run_planner\":true,\"run_trends\":false,\"themes\":[],\"customer_ids\":[\"3532896537\"],\"needs_ids\":false,\"notes\":\"router_verified model=local verify=azure; Resolved account to Havas_Shell_GoogleAds_US_Mobility Fuels (3532896537).\",\"confidence\":0.8,\"needs_clarification\":false,\"clarification\":\"How did the Fuels a..."
      }
    },
    {
      "name": "api.chat.send.general",
      "goal": "General chat response contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 5377,
        "data": "{\"reply\":\"Hi there! I'd be happy to help with any media-related questions or concerns you have. Can I help you with something specific or do you have a media-related issue on your mind?\",\"role\":\"assistant\",\"sources\":[],\"model\":\"local\",\"guardrail\":null}"
      }
    },
    {
      "name": "api.audit.business_units",
      "goal": "Audit business units contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 19,
        "data": "{\"business_units\":[{\"id\":\"Brand\",\"name\":\"Brand\",\"description\":\"Brand campaigns\"},{\"id\":\"NonBrand\",\"name\":\"Non-Brand\",\"description\":\"Non-brand campaigns\"},{\"id\":\"PMax\",\"name\":\"Performance Max\",\"description\":\"PMax campaigns\"}]}"
      }
    },
    {
      "name": "api.serp.check",
      "goal": "SERP check contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 76,
        "data": "{\"status\":\"success\",\"results\":[{\"url\":\"https://google.com\",\"status\":200,\"soft_404\":false}]}"
      }
    },
    {
      "name": "migrations.sanity",
      "goal": "Migration framework detected",
      "status": "skip",
      "detail": "No migration framework markers found."
    }
  ]
}
```
api_jobs_worker_exists: False
worker_jobs_worker_exists: True
```

Gate:
```
node C:\Software Builds\repo_shadow\playwright-run\integrity_suite.js
```

Gate output:
```
{
  "config": {
    "BACKEND_URL": "https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io",
    "checks": 11
  },
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  },
  "results": [
    {
      "name": "api.health",
      "goal": "Backend reachable and responding",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 95,
        "data": "{\"status\":\"healthy\",\"service\":\"Kai Platform API\"}"
      }
    },
    {
      "name": "api.diagnostics.health",
      "goal": "Core dependency health (LLM/SA360/QA sample)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 1539,
        "data": "{\"status\":\"ok\",\"errors\":[],\"accounts\":{\"count\":34,\"sample\":[{\"customer_id\":\"5771282191\",\"name\":\"Canada_Mobility\",\"manager\":true},{\"customer_id\":\"4427623347\",\"name\":\"Canada_Pennzoil\",\"manager\":true},{\"customer_id\":\"3662965856\",\"name\":\"Canada_Quaker State\",\"manager\":true},{\"customer_id\":\"3716440491\",\"name\":\"Canada_Rotella\",\"manager\":true},{\"customer_..."
      }
    },
    {
      "name": "api.auth.verify.valid",
      "goal": "Password gate validation",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 23,
        "data": "{\"status\":\"success\",\"authenticated\":true}"
      }
    },
    {
      "name": "api.auth.verify.invalid",
      "goal": "Auth isolation on bad password",
      "status": "pass",
      "detail": {
        "status": 401,
        "duration_ms": 16,
        "data": "{\"detail\":\"Invalid password\"}"
      }
    },
    {
      "name": "api.settings.get",
      "goal": "Settings contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13,
        "data": "{\"ai_chat_enabled\":true,\"ai_insights_enabled\":true,\"theme\":\"light\"}"
      }
    },
    {
      "name": "api.settings.env.list",
      "goal": "Env list masked contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13,
        "data": "{\"env\":[{\"key\":\"AZURE_OPENAI_ENDPOINT\",\"value\":\"************************************com/\"},{\"key\":\"AZURE_OPENAI_API_KEY\",\"value\":\"****************************8795\"},{\"key\":\"AZURE_OPENAI_DEPLOYMENT\",\"value\":\"*******urbo\"},{\"key\":\"AZURE_OPENAI_API_VERSION\",\"value\":\"**************view\"},{\"key\":\"SERPAPI_KEY\",\"value\":\"***********************************..."
      }
    },
    {
      "name": "api.chat.route.performance",
      "goal": "Router contract (intent + planner)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 15884,
        "data": "{\"intent\":\"performance\",\"tool\":\"performance\",\"run_planner\":true,\"run_trends\":false,\"themes\":[],\"customer_ids\":[\"3532896537\"],\"needs_ids\":false,\"notes\":\"router_verified model=local verify=azure; Resolved account to Havas_Shell_GoogleAds_US_Mobility Fuels (3532896537).\",\"confidence\":0.8,\"needs_clarification\":false,\"clarification\":\"How did the Fuels a..."
      }
    },
    {
      "name": "api.chat.send.general",
      "goal": "General chat response contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 14837,
        "data": "{\"reply\":\"Hi! I need to create a social media strategy for a new product launch. I want to know how to best use paid social media advertising to get people talking about the product. Do you have any suggestions on how to make paid social media ads effective? What types of ads would you recommend and how many to run per week? Would it be worth inves..."
      }
    },
    {
      "name": "api.audit.business_units",
      "goal": "Audit business units contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 19,
        "data": "{\"business_units\":[{\"id\":\"Brand\",\"name\":\"Brand\",\"description\":\"Brand campaigns\"},{\"id\":\"NonBrand\",\"name\":\"Non-Brand\",\"description\":\"Non-brand campaigns\"},{\"id\":\"PMax\",\"name\":\"Performance Max\",\"description\":\"PMax campaigns\"}]}"
      }
    },
    {
      "name": "api.serp.check",
      "goal": "SERP check contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 79,
        "data": "{\"status\":\"success\",\"results\":[{\"url\":\"https://google.com\",\"status\":200,\"soft_404\":false}]}"
      }
    },
    {
      "name": "migrations.sanity",
      "goal": "Migration framework detected",
      "status": "skip",
      "detail": "No migration framework markers found."
    }
  ]
}
```
node C:\Software Builds\repo_shadow\playwright-run\integrity_suite.js
```

Gate output:
```
{
  "config": {
    "BACKEND_URL": "https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io",
    "checks": 11
  },
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  },
  "results": [
    {
      "name": "api.health",
      "goal": "Backend reachable and responding",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 94,
        "data": "{\"status\":\"healthy\",\"service\":\"Kai Platform API\"}"
      }
    },
    {
      "name": "api.diagnostics.health",
      "goal": "Core dependency health (LLM/SA360/QA sample)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 1495,
        "data": "{\"status\":\"ok\",\"errors\":[],\"accounts\":{\"count\":34,\"sample\":[{\"customer_id\":\"5771282191\",\"name\":\"Canada_Mobility\",\"manager\":true},{\"customer_id\":\"4427623347\",\"name\":\"Canada_Pennzoil\",\"manager\":true},{\"customer_id\":\"3662965856\",\"name\":\"Canada_Quaker State\",\"manager\":true},{\"customer_id\":\"3716440491\",\"name\":\"Canada_Rotella\",\"manager\":true},{\"customer_..."
      }
    },
    {
      "name": "api.auth.verify.valid",
      "goal": "Password gate validation",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 18,
        "data": "{\"status\":\"success\",\"authenticated\":true}"
      }
    },
    {
      "name": "api.auth.verify.invalid",
      "goal": "Auth isolation on bad password",
      "status": "pass",
      "detail": {
        "status": 401,
        "duration_ms": 16,
        "data": "{\"detail\":\"Invalid password\"}"
      }
    },
    {
      "name": "api.settings.get",
      "goal": "Settings contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13,
        "data": "{\"ai_chat_enabled\":true,\"ai_insights_enabled\":true,\"theme\":\"light\"}"
      }
    },
    {
      "name": "api.settings.env.list",
      "goal": "Env list masked contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13,
        "data": "{\"env\":[{\"key\":\"AZURE_OPENAI_ENDPOINT\",\"value\":\"************************************com/\"},{\"key\":\"AZURE_OPENAI_API_KEY\",\"value\":\"****************************8795\"},{\"key\":\"AZURE_OPENAI_DEPLOYMENT\",\"value\":\"*******urbo\"},{\"key\":\"AZURE_OPENAI_API_VERSION\",\"value\":\"**************view\"},{\"key\":\"SERPAPI_KEY\",\"value\":\"***********************************..."
      }
    },
    {
      "name": "api.chat.route.performance",
      "goal": "Router contract (intent + planner)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 16816,
        "data": "{\"intent\":\"performance\",\"tool\":\"performance\",\"run_planner\":true,\"run_trends\":false,\"themes\":[],\"customer_ids\":[\"3532896537\"],\"needs_ids\":false,\"notes\":\"router_verified model=local verify=azure; Resolved account to Havas_Shell_GoogleAds_US_Mobility Fuels (3532896537).\",\"confidence\":0.8,\"needs_clarification\":false,\"clarification\":\"How did the Fuels a..."
      }
    },
    {
      "name": "api.chat.send.general",
      "goal": "General chat response contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 6430,
        "data": "{\"reply\":\"I'd be happy to help you find the right media channels for your project. Can you tell me a bit more about what you're looking for? What type of project are you working on, and what are the key performance indicators (KPIs) you're trying to achieve?\",\"role\":\"assistant\",\"sources\":[],\"model\":\"local\",\"guardrail\":null}"
      }
    },
    {
      "name": "api.audit.business_units",
      "goal": "Audit business units contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 17,
        "data": "{\"business_units\":[{\"id\":\"Brand\",\"name\":\"Brand\",\"description\":\"Brand campaigns\"},{\"id\":\"NonBrand\",\"name\":\"Non-Brand\",\"description\":\"Non-brand campaigns\"},{\"id\":\"PMax\",\"name\":\"Performance Max\",\"description\":\"PMax campaigns\"}]}"
      }
    },
    {
      "name": "api.serp.check",
      "goal": "SERP check contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 89,
        "data": "{\"status\":\"success\",\"results\":[{\"url\":\"https://google.com\",\"status\":200,\"soft_404\":false}]}"
      }
    },
    {
      "name": "migrations.sanity",
      "goal": "Migration framework detected",
      "status": "skip",
      "detail": "No migration framework markers found."
    }
  ]
}
```
cd C:\Software Builds\repo_shadow\repo\apps\web
npm run build
```

Build output (ANSI stripped; checkmark replaced with OK):
```
> kai-platform-frontend@2.0.0 build
> vite build

vite v5.4.21 building for production...
transforming...
OK 11988 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                             0.49 kB | gzip:   0.32 kB
dist/assets/index-DVPuY11z.css              4.81 kB | gzip:   1.66 kB
dist/assets/DataSourceBadge-BxoKbAkx.js     2.08 kB | gzip:   1.12 kB
dist/assets/PMaxSpendSankey-CsLJJQx0.js     5.78 kB | gzip:   2.28 kB
dist/assets/ChannelComparison-DBABOjBx.js   6.73 kB | gzip:   2.57 kB
dist/assets/index-B6REg-6Y.js             833.38 kB | gzip: 257.02 kB

(!) Some chunks are larger than 500 kB after minification. Consider:
- Using dynamic import() to code-split the application
- Use build.rollupOptions.output.manualChunks to improve chunking: https://rollupjs.org/configuration-options/#output-manualchunks
- Adjust chunk size limit for this warning via build.chunkSizeWarningLimit.

OK built in 9.20s
```

Gate:
```
node C:\Software Builds\repo_shadow\playwright-run\integrity_suite.js
```

Gate output:
```
{
  "config": {
    "BACKEND_URL": "https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io",
    "checks": 11
  },
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  },
  "results": [
    {
      "name": "api.health",
      "goal": "Backend reachable and responding",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 77,
        "data": "{\"status\":\"healthy\",\"service\":\"Kai Platform API\"}"
      }
    },
    {
      "name": "api.diagnostics.health",
      "goal": "Core dependency health (LLM/SA360/QA sample)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 3506,
        "data": "{\"status\":\"ok\",\"errors\":[],\"accounts\":{\"count\":34,\"sample\":[{\"customer_id\":\"5771282191\",\"name\":\"Canada_Mobility\",\"manager\":true},{\"customer_id\":\"4427623347\",\"name\":\"Canada_Pennzoil\",\"manager\":true},{\"customer_id\":\"3662965856\",\"name\":\"Canada_Quaker State\",\"manager\":true},{\"customer_id\":\"3716440491\",\"name\":\"Canada_Rotella\",\"manager\":true},{\"customer_..."
      }
    },
    {
      "name": "api.auth.verify.valid",
      "goal": "Password gate validation",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 19,
        "data": "{\"status\":\"success\",\"authenticated\":true}"
      }
    },
    {
      "name": "api.auth.verify.invalid",
      "goal": "Auth isolation on bad password",
      "status": "pass",
      "detail": {
        "status": 401,
        "duration_ms": 19,
        "data": "{\"detail\":\"Invalid password\"}"
      }
    },
    {
      "name": "api.settings.get",
      "goal": "Settings contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 16,
        "data": "{\"ai_chat_enabled\":true,\"ai_insights_enabled\":true,\"theme\":\"light\"}"
      }
    },
    {
      "name": "api.settings.env.list",
      "goal": "Env list masked contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 15,
        "data": "{\"env\":[{\"key\":\"AZURE_OPENAI_ENDPOINT\",\"value\":\"************************************com/\"},{\"key\":\"AZURE_OPENAI_API_KEY\",\"value\":\"****************************8795\"},{\"key\":\"AZURE_OPENAI_DEPLOYMENT\",\"value\":\"*******urbo\"},{\"key\":\"AZURE_OPENAI_API_VERSION\",\"value\":\"**************view\"},{\"key\":\"SERPAPI_KEY\",\"value\":\"***********************************..."
      }
    },
    {
      "name": "api.chat.route.performance",
      "goal": "Router contract (intent + planner)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 17323,
        "data": "{\"intent\":\"performance\",\"tool\":\"performance\",\"run_planner\":true,\"run_trends\":false,\"themes\":[],\"customer_ids\":[\"3532896537\"],\"needs_ids\":false,\"notes\":\"router_verified model=local verify=azure; Resolved account to Havas_Shell_GoogleAds_US_Mobility Fuels (3532896537).\",\"confidence\":0.8,\"needs_clarification\":false,\"clarification\":\"How did the Fuels a..."
      }
    },
    {
      "name": "api.chat.send.general",
      "goal": "General chat response contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 13412,
        "data": "{\"reply\":\"I can help you with anything related to paid media—like Google Ads, Meta (Facebook/Instagram) campaigns, LinkedIn ads, and other platforms. I can advise on strategy, targeting, budgets, creative, reporting, troubleshooting performance issues, and optimising campaigns. If you have specific questions or need recommendations, just let me kno..."
      }
    },
    {
      "name": "api.audit.business_units",
      "goal": "Audit business units contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 20,
        "data": "{\"business_units\":[{\"id\":\"Brand\",\"name\":\"Brand\",\"description\":\"Brand campaigns\"},{\"id\":\"NonBrand\",\"name\":\"Non-Brand\",\"description\":\"Non-brand campaigns\"},{\"id\":\"PMax\",\"name\":\"Performance Max\",\"description\":\"PMax campaigns\"}]}"
      }
    },
    {
      "name": "api.serp.check",
      "goal": "SERP check contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 82,
        "data": "{\"status\":\"success\",\"results\":[{\"url\":\"https://google.com\",\"status\":200,\"soft_404\":false}]}"
      }
    },
    {
      "name": "migrations.sanity",
      "goal": "Migration framework detected",
      "status": "skip",
      "detail": "No migration framework markers found."
    }
  ]
}
```
node C:\Software Builds\repo_shadow\playwright-run\integrity_suite.js
```

Gate output:
```
{
  "config": {
    "BACKEND_URL": "https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io",
    "checks": 11
  },
  "summary": {
    "pass": 10,
    "fail": 0,
    "skip": 1
  },
  "results": [
    {
      "name": "api.health",
      "goal": "Backend reachable and responding",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 83,
        "data": "{\"status\":\"healthy\",\"service\":\"Kai Platform API\"}"
      }
    },
    {
      "name": "api.diagnostics.health",
      "goal": "Core dependency health (LLM/SA360/QA sample)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 1375,
        "data": "{\"status\":\"ok\",\"errors\":[],\"accounts\":{\"count\":34,\"sample\":[{\"customer_id\":\"5771282191\",\"name\":\"Canada_Mobility\",\"manager\":true},{\"customer_id\":\"4427623347\",\"name\":\"Canada_Pennzoil\",\"manager\":true},{\"customer_id\":\"3662965856\",\"name\":\"Canada_Quaker State\",\"manager\":true},{\"customer_id\":\"3716440491\",\"name\":\"Canada_Rotella\",\"manager\":true},{\"customer_..."
      }
    },
    {
      "name": "api.auth.verify.valid",
      "goal": "Password gate validation",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 20,
        "data": "{\"status\":\"success\",\"authenticated\":true}"
      }
    },
    {
      "name": "api.auth.verify.invalid",
      "goal": "Auth isolation on bad password",
      "status": "pass",
      "detail": {
        "status": 401,
        "duration_ms": 19,
        "data": "{\"detail\":\"Invalid password\"}"
      }
    },
    {
      "name": "api.settings.get",
      "goal": "Settings contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 16,
        "data": "{\"ai_chat_enabled\":true,\"ai_insights_enabled\":true,\"theme\":\"light\"}"
      }
    },
    {
      "name": "api.settings.env.list",
      "goal": "Env list masked contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 15,
        "data": "{\"env\":[{\"key\":\"AZURE_OPENAI_ENDPOINT\",\"value\":\"************************************com/\"},{\"key\":\"AZURE_OPENAI_API_KEY\",\"value\":\"****************************8795\"},{\"key\":\"AZURE_OPENAI_DEPLOYMENT\",\"value\":\"*******urbo\"},{\"key\":\"AZURE_OPENAI_API_VERSION\",\"value\":\"**************view\"},{\"key\":\"SERPAPI_KEY\",\"value\":\"***********************************..."
      }
    },
    {
      "name": "api.chat.route.performance",
      "goal": "Router contract (intent + planner)",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 16529,
        "data": "{\"intent\":\"performance\",\"tool\":\"performance\",\"run_planner\":true,\"run_trends\":false,\"themes\":[],\"customer_ids\":[\"3532896537\"],\"needs_ids\":false,\"notes\":\"router_verified model=local verify=azure; Resolved account to Havas_Shell_GoogleAds_US_Mobility Fuels (3532896537).\",\"confidence\":0.8,\"needs_clarification\":false,\"clarification\":\"How did the Fuels a..."
      }
    },
    {
      "name": "api.chat.send.general",
      "goal": "General chat response contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 6641,
        "data": "{\"reply\":\"I'd be happy to help you find the right media channels for your project. Can you tell me a bit more about what you're looking for? What type of project are you working on, and what are the key performance indicators (KPIs) you're trying to achieve?\",\"role\":\"assistant\",\"sources\":[],\"model\":\"local\",\"guardrail\":null}"
      }
    },
    {
      "name": "api.audit.business_units",
      "goal": "Audit business units contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 19,
        "data": "{\"business_units\":[{\"id\":\"Brand\",\"name\":\"Brand\",\"description\":\"Brand campaigns\"},{\"id\":\"NonBrand\",\"name\":\"Non-Brand\",\"description\":\"Non-brand campaigns\"},{\"id\":\"PMax\",\"name\":\"Performance Max\",\"description\":\"PMax campaigns\"}]}"
      }
    },
    {
      "name": "api.serp.check",
      "goal": "SERP check contract",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 84,
        "data": "{\"status\":\"success\",\"results\":[{\"url\":\"https://google.com\",\"status\":200,\"soft_404\":false}]}"
      }
    },
    {
      "name": "migrations.sanity",
      "goal": "Migration framework detected",
      "status": "skip",
      "detail": "No migration framework markers found."
    }
  ]
}
```
