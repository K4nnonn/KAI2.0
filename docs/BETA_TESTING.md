# Kai Broad Beta Testing Guide

This guide is for beta testers validating Kai end-to-end (SA360 OAuth -> account selection -> column browsing -> chat analysis).

## URLs

- Frontend (web UI): `https://kai-web.happyrock-0c0fe8c1.eastus.azurecontainerapps.io`
- Backend (API): `https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io`

## Getting Started (Required)

1. Open the web UI.
1. Authenticate:
   - If Microsoft SSO is enabled in the UI, prefer **Sign in with Microsoft**.
   - Otherwise use the password gate (provided by the admin).
1. In **Kai Chat**, click **Connect SA360** and complete Google OAuth.
1. Set the **Manager ID (MCC)** and click **Save MCC**.
   - Expected: a success toast and a visible **Account (by name)** picker.
1. Choose an **Account (by name)** from the picker.
   - Expected: the selection is remembered for your session and used for Performance tools.

## How To Validate SA360 Data

### Browse columns (conversion actions)

1. In Kai Chat header, click **Browse columns**.
1. Select an account.
1. Use search to find a conversion action/metric (e.g. `Store visits`, `Directions`, or a client-specific action).
1. Click **Use in chat** to copy a ready prompt and jump back to Kai Chat.

### Example questions (copy/paste)

- `Why did Store visits change week over week? Which campaigns drove it?`
- `Why did Local actions - Directions change week over week? Which devices drove it?`
- `Why did FR_Intent_clicks change week over week? Which campaigns drove it?`
  - Note: Kai should infer the closest matching conversion action from your account's catalog.

## What To Capture When Reporting Bugs

- The **exact prompt** you typed.
- Whether **SA360 connected** is shown.
- The selected **MCC** and **Account (by name)** (name is enough; IDs optional).
- Approximate response time (e.g. `~5s`, `~30s`).
- Screenshots of any UI banner/errors.

## Optional: QA Harness (Engineering)

Engineers can run a regression sweep against the API:

```powershell
Set-Location "C:\\Users\\kelvin.le\\OneDrive - Havas\\Desktop\\Builds\\Paid_Search_Apps\\Paid_Search_Apps-main"

powershell -ExecutionPolicy Bypass -File scripts/verification/full_qa.ps1 `
  -Backend "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io" `
  -SessionId "<YOUR_SESSION_ID>" `
  -TargetCustomerId "<CUSTOMER_ID>" `
  -CustomMetric "FR_Intent_clicks" `
  -SkipEnv
```

Notes:
- `SessionId` comes from the browser (DevTools -> `sessionStorage.getItem('kai_chat_session_id')`).
- The script writes artifacts under `verification_runs/` inside this repo.

## Optional: UI E2E (Playwright)

Engineers can run a reproducible UI+API regression sweep (writes artifacts under `verification_runs/ui_e2e_*`):

```powershell
Set-Location "C:\\Users\\kelvin.le\\OneDrive - Havas\\Desktop\\Builds\\Paid_Search_Apps\\Paid_Search_Apps-main"

$env:FRONTEND_URL = "https://kai-web.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
$env:BACKEND_URL = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
$env:KAI_SESSION_ID = "<YOUR_SESSION_ID>"

powershell -ExecutionPolicy Bypass -File scripts/verification/ui_e2e.ps1
```
