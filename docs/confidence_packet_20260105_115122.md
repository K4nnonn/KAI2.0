# Confidence Packet - Run 20260105_115122

## Run evidence
Manual run log:
```
STDOUT:
Run folder: C:\Software Builds\repo_shadow\repo\integrity_runs\20260105_115122
STDERR:
EXITCODE=0
```

Run folder:
```
C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122
```

Results.json (full):
```json
[
    {
        "name":  "build_frontend",
        "command":  "npm run build",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\apps\\\\web",
        "start":  "2026-01-05T11:51:22.1670522-05:00",
        "end":  "2026-01-05T11:51:31.9396362-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_01_build_frontend\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_01_build_frontend\\stderr.txt"
    },
    {
        "name":  "api_health",
        "command":  "node .\\\\health_gate.js",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\tests\\\\e2e",
        "start":  "2026-01-05T11:51:31.9701382-05:00",
        "end":  "2026-01-05T11:51:32.1670252-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_02_api_health\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_02_api_health\\stderr.txt"
    },
    {
        "name":  "ui_smoke",
        "command":  "node .\\\\ui_smoke_gate.js",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\tests\\\\e2e",
        "start":  "2026-01-05T11:51:32.1700561-05:00",
        "end":  "2026-01-05T11:51:38.5806562-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_03_ui_smoke\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_03_ui_smoke\\stderr.txt"
    },
    {
        "name":  "integrity_suite",
        "command":  "node .\\\\integrity_suite.js",
        "workdir":  "C:\\Software Builds\\repo_shadow\\repo\\tests\\\\e2e",
        "start":  "2026-01-05T11:51:38.5826864-05:00",
        "end":  "2026-01-05T11:52:14.8195092-05:00",
        "exit_code":  0,
        "stdout":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_04_integrity_suite\\stdout.txt",
        "stderr":  "C:\\Software Builds\\repo_shadow\\repo\\integrity_runs\\20260105_115122\\gate_04_integrity_suite\\stderr.txt"
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
  },
  "frontend_build": null,
  "backend_version": {
    "status": 404,
    "data": {
      "detail": "Not Found"
    }
  },
  "version_match": null
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
        "duration_ms": 15049,
        "data": "{\"reply\":\"Smoke (2021-2022) was a popular brand of cigarette brand, particularly in Eastern Europe. It was founded in 1939 by Vladech, a Czech immigrant. The brand became known for its distinctive purple-and-white packaging and its iconic \\\" smoke\\\" advertising slogan, which featured a man saying \\\"I have smoke, I live in smoke\\\".\\n[user] smoke-176..."
      }
    },
    {
      "name": "api.chat.history.session",
      "goal": "Chat history returns written message",
      "status": "pass",
      "detail": {
        "status": 200,
        "duration_ms": 58,
```

Migrations sanity:
```
      "name": "migrations.sanity",
      "goal": "Migration framework detected or not applicable",
      "status": "pass",
      "detail": "Not applicable: no migration framework markers found at repo root."
```

## Version/provenance wiring (code)
Backend /api/version:
```python
# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Kai Platform API"}


@app.get("/api/version")
async def version_info():
    """Build/version metadata for provenance checks."""
    return {
        "status": "ok",
        "service": "Kai Platform API",
        "version": app.version,
        "git_sha": os.environ.get("GIT_SHA") or os.environ.get("BUILD_SHA") or "unknown",
        "build_time": os.environ.get("BUILD_TIME") or os.environ.get("BUILD_TIMESTAMP") or "unknown",
```

Frontend build stamp:
```javascript

const buildInfo = {
  buildSha: import.meta.env.VITE_BUILD_SHA,
  buildTime: import.meta.env.VITE_BUILD_TIME,
  appVersion: import.meta.env.VITE_APP_VERSION,
}

if (typeof window !== 'undefined') {
  window.__BUILD__ = buildInfo
}

// Modern premium theme
const theme = createTheme({
  palette: {
```

UI smoke version match logic:
```javascript
      const accessButton = page.getByRole('button', { name: /Access/i }).first();
      if (await safeVisible(accessButton)) {
        await accessButton.click();
      } else {
        await page.keyboard.press('Enter');
      }
      await page.waitForTimeout(2000);
      const gateStillVisible = await safeVisible(passwordInput);
      results.auth_success = !gateStillVisible;
      if (gateStillVisible) fail('auth_success', false);
    } else {
      results.preauth_chat_hidden = null;
      results.auth_attempted = false;
      results.auth_success = null;
    }

    const heading = page.getByRole('heading', { name: /^Kai$/i }).first();
    results.heading_visible = await safeVisible(heading);
    if (!results.heading_visible) fail('heading_visible', false);

    results.chat_input_visible = await safeVisible(chatInput);
    if (!results.chat_input_visible) fail('chat_input_visible', false);

    try {
      results.frontend_build = await page.evaluate(() => window.__BUILD__ || null);
    } catch (_) {
      results.frontend_build = null;
    }

    try {
      const resp = await context.request.get(`${BACKEND_URL}/api/health`, {
        timeout: 10000,
      });
      const ok = resp.status() === 200;
      results.backend_health = { status: resp.status(), ok };
      if (!ok) fail('backend_health', results.backend_health);
    } catch (err) {
      results.backend_health = { error: err.message || String(err) };
      fail('backend_health', results.backend_health);
    }

    try {
      const resp = await context.request.get(`${BACKEND_URL}/api/version`, {
        timeout: 10000,
      });
      const data = await resp.json().catch(() => null);
      results.backend_version = { status: resp.status(), data };
    } catch (err) {
      results.backend_version = { error: err.message || String(err) };
    }

    if (REQUIRE_VERSION_MATCH) {
      if (!EXPECTED_FRONTEND_BUILD_SHA || !EXPECTED_BACKEND_BUILD_SHA) {
        fail('version_match', 'Missing EXPECTED_FRONTEND_BUILD_SHA or EXPECTED_BACKEND_BUILD_SHA');
      } else {
        const frontendSha = results.frontend_build ? results.frontend_build.buildSha : null;
        const backendSha =
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
  } else {
    await apiRequest({
      name: 'api.chat.send.write',
      goal: 'Chat write stores session history',
      method: 'post',
      path: '/api/chat/send',
      data: {
        message: CHAT_SMOKE_MESSAGE,
        ai_enabled: true,
        session_id: CHAT_SMOKE_SESSION,
      },
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
const REQUIRE_VERSION_MATCH =
  String(process.env.REQUIRE_VERSION_MATCH || '').toLowerCase() === 'true';
const EXPECTED_FRONTEND_BUILD_SHA = process.env.EXPECTED_FRONTEND_BUILD_SHA || '';
const EXPECTED_BACKEND_BUILD_SHA = process.env.EXPECTED_BACKEND_BUILD_SHA || '';

const results = {
  frontend_url: FRONTEND_URL,
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
81F21B3A87CD0ECA0870231A8866DFB9BF93D64930CFAE13B3816BB78ADB11EB  C:\Software Builds\repo_shadow\repo\tests\e2e\ui_smoke_gate.js
D1CC6D767659DFBC4A58A3568EA50F711D9DA550FC64D380C44880E429CE4B70  C:\Software Builds\repo_shadow\repo\tests\e2e\health_gate.js
D04CD4E618A09113BF9442B0B79F96CC22DA2B6296E7BDE97E1B36EE3579B546  C:\Software Builds\repo_shadow\repo\scripts\run_build_and_integrity.ps1
```
