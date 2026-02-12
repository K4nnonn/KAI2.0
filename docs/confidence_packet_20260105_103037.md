# Confidence Packet - Run 20260105_103037

## Run evidence
Manual run log:
```
STDOUT:
Run folder: C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_103037
STDERR:
STDERR:
EXITCODE=0
```

Run folder:
```
C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037
```

Results.json (full):
```json
[
    {
        "name":  "build_frontend",
        "command":  "npm run build",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\apps\\\\web",
        "start":  "2026-01-05T10:30:37.1353364-05:00",
        "end":  "2026-01-05T10:30:46.5970475-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_01_build_frontend\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_01_build_frontend\\stderr.txt"
    },
    {
        "name":  "api_health",
        "command":  "node .\\\\health_gate.js",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\tests\\\\e2e",
        "start":  "2026-01-05T10:30:46.6261813-05:00",
        "end":  "2026-01-05T10:30:46.8265388-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_02_api_health\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_02_api_health\\stderr.txt"
    },
    {
        "name":  "ui_smoke",
        "command":  "node .\\\\ui_smoke_gate.js",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\tests\\\\e2e",
        "start":  "2026-01-05T10:30:46.8285432-05:00",
        "end":  "2026-01-05T10:30:50.1734883-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_03_ui_smoke\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_03_ui_smoke\\stderr.txt"
    },
    {
        "name":  "integrity_suite",
        "command":  "node .\\\\integrity_suite.js",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\tests\\\\e2e",
        "start":  "2026-01-05T10:30:50.1748497-05:00",
        "end":  "2026-01-05T10:31:14.3660165-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_04_integrity_suite\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_103037\\gate_04_integrity_suite\\stderr.txt"
    }
]
```

## UI smoke output
```json
{
  "frontend_url": "https://www.kelvinale.com",
  "backend_url": "https://kai-platform-backend.agreeablehill-69579b55.eastus.azurecontainerapps.io",
  "gate_present": true,
  "preauth_chat_hidden": true,
  "auth_attempted": true,
  "auth_success": true,
  "heading_visible": true,
  "chat_input_visible": true,
  "backend_health": {
    "status": 200,
    "ok": true
  }
}
```

## Integrity suite summary + key checks
Summary:
```
  "summary": {
    "pass": 15,
    "fail": 0,
    "skip": 0
  },
```

Chat write + history:
```
      "name": "api.chat.send.write",
      "goal": "Chat write stores session history",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 1891,
        "data": "{\"reply\":\"I think I have the answer. Can you confirm if the media company, Smoke, is a well-known or reputable name in the media industry?\",\"role\":\"assistant\",\"sources\":[],\"model\":\"local\",\"guardrail\":null}"
      }
    },
    {
      "name": "api.chat.history.session",
      "goal": "Chat history returns written message",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 65,
```

Migrations sanity:
```
      "name": "migrations.sanity",
      "goal": "Migration framework detected or not applicable",
      "status": "pass",
      "detail": "Not applicable: no migration framework markers found at repo root."
```

## Safety controls (env required + write guard)
Integrity suite env requirements:
```javascript
const requireEnv = (name) => {
  const value = process.env[name];
  if (!value) {
    console.error(`Missing required env var: ${name}`);
    process.exit(1);
  }
  return value;
};

const BACKEND_URL = requireEnv('BACKEND_URL');
const ACCESS_PASSWORD = requireEnv('KAI_ACCESS_PASSWORD');
const ALLOW_WRITE_SMOKE = String(process.env.ALLOW_WRITE_SMOKE || '').toLowerCase() === 'true';
const ALLOW_PROD_WRITE_SMOKE =
  String(process.env.ALLOW_PROD_WRITE_SMOKE || '').toLowerCase() === 'true';

const isProdBackend = (url) => {
  try {
    const host = new URL(url).hostname;
    if (/kelvinale\.com$/i.test(host)) return true;
    if (/agreeablehill-69579b55/i.test(host)) return true;
    return false;
  } catch (_) {
    return false;
  }
};

const writeBlockedReason = () => {
  if (!ALLOW_WRITE_SMOKE) {
    return 'ALLOW_WRITE_SMOKE not enabled.';
  }
  if (isProdBackend(BACKEND_URL) && !ALLOW_PROD_WRITE_SMOKE) {
    return 'Write smoke blocked on prod. Set ALLOW_PROD_WRITE_SMOKE=true to override.';
  }
  return '';
};
```

Integrity suite write guard:
```javascript
  const blockedReason = writeBlockedReason();
  if (blockedReason) {
    record({
      name: 'api.chat.send.write',
      goal: 'Chat write stores session history',
      status: 'skip',
      detail: blockedReason,
    });
    record({
      name: 'api.chat.history.session',
      goal: 'Chat history returns written message',
      status: 'skip',
      detail: blockedReason,
    });
```

UI smoke env requirements:
```javascript
const { chromium } = require('playwright');

const requireEnv = (name) => {
  const value = process.env[name];
  if (!value) {
    console.error(`Missing required env var: ${name}`);
    process.exit(1);
  }
  return value;
};

const FRONTEND_URL = requireEnv('FRONTEND_URL');
const BACKEND_URL = requireEnv('BACKEND_URL');
const ACCESS_PASSWORD = requireEnv('KAI_ACCESS_PASSWORD');
```

Health gate env requirement:
```javascript
const axios = require('axios');

const BACKEND_URL = process.env.BACKEND_URL;
if (!BACKEND_URL) {
  console.error('Missing required env var: BACKEND_URL');
  process.exit(1);
}
```

## Script hashes (SHA256)
```
4F92435E20CBB600DA9127989E4A398175CFD71937FE0B7204B1A63EFCAF9D43  C:\Software Builds\repo_shadow\repo\tests\e2e\integrity_suite.js
882AC4EC843FD87B856093F544B8FDA6F7E43F472906B931821F64CA55CA1398  C:\Software Builds\repo_shadow\repo\tests\e2e\ui_smoke_gate.js
D1CC6D767659DFBC4A58A3568EA50F711D9DA550FC64D380C44880E429CE4B70  C:\Software Builds\repo_shadow\repo\tests\e2e\health_gate.js
D04CD4E618A09113BF9442B0B79F96CC22DA2B6296E7BDE97E1B36EE3579B546  C:\Software Builds\repo_shadow\repo\scripts\run_build_and_integrity.ps1
```
