param(
  [string]$Frontend = $env:FRONTEND_URL,
  [string]$Backend = $env:BACKEND_URL,
  [string]$SessionId = $env:KAI_SESSION_ID,
  [int]$RouteBudgetSec = 3,
  [int]$SendBudgetSec = 15,
  [string]$Browser = "chromium"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Frontend)) {
  $Frontend = "https://kai-web.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}
if ([string]::IsNullOrWhiteSpace($Backend)) {
  $Backend = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$RunBase = Join-Path $RepoRoot "verification_runs"
New-Item -ItemType Directory -Force -Path $RunBase | Out-Null
$RunDir = Join-Path $RunBase ("ui_e2e_" + $Stamp)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Write-Json($Path, $Obj) {
  $Obj | ConvertTo-Json -Depth 8 | Set-Content -Path $Path -Encoding UTF8
}

$WebRoot = Join-Path $RepoRoot "apps\\web"
if (-not (Test-Path $WebRoot)) {
  throw "apps/web not found at: $WebRoot"
}

# Pass env vars to the Playwright tests (UI + API checks).
$env:FRONTEND_URL = $Frontend
$env:BACKEND_URL = $Backend
$env:KAI_SESSION_ID = $SessionId
$env:KAI_ROUTE_BUDGET_SEC = [string]$RouteBudgetSec
$env:KAI_SEND_BUDGET_SEC = [string]$SendBudgetSec

# Keep artifacts inside the repo (share-safe). Avoid writing to random user folders on shared machines.
$env:PLAYWRIGHT_REPORT_DIR = (Join-Path $RunDir "playwright-report")
$env:PLAYWRIGHT_OUTPUT_DIR = (Join-Path $RunDir "test-results")

$logPath = Join-Path $RunDir "playwright_output.txt"
$exitCode = 0

Push-Location $WebRoot
try {
  # Use the repo-local playwright binary to avoid PowerShell execution policy issues with npx.ps1.
  & .\\node_modules\\.bin\\playwright.cmd test --browser=$Browser 2>&1 | Tee-Object -FilePath $logPath
  $exitCode = $LASTEXITCODE
} catch {
  $_ | Out-String | Add-Content -Path $logPath
  $exitCode = 1
} finally {
  Pop-Location
}

$summary = [ordered]@{
  run_id = $Stamp
  kind = "ui_e2e"
  frontend = $Frontend
  backend = $Backend
  browser = $Browser
  route_budget_sec = $RouteBudgetSec
  send_budget_sec = $SendBudgetSec
  session_id_present = (-not [string]::IsNullOrWhiteSpace($SessionId))
  exit_code = $exitCode
  passed = ($exitCode -eq 0)
  artifacts = [ordered]@{
    run_dir = $RunDir
    playwright_report = $env:PLAYWRIGHT_REPORT_DIR
    test_results = $env:PLAYWRIGHT_OUTPUT_DIR
    log = $logPath
  }
  timestamp = (Get-Date).ToString("o")
}

Write-Json (Join-Path $RunDir "summary.json") $summary
Write-Output "Run folder: $RunDir"

if ($exitCode -ne 0) { exit $exitCode }

