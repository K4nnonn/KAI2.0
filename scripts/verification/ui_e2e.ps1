param(
  [string]$Frontend = $env:FRONTEND_URL,
  [string]$Backend = $env:BACKEND_URL,
  [string]$SessionId = $env:KAI_SESSION_ID,
  [int]$RouteBudgetSec = 3,
  [int]$SendBudgetSec = 15,
  [string]$Browser = "chromium",
  [switch]$AllowNoSession
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Frontend)) {
  $Frontend = "https://kai-web.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}
if ([string]::IsNullOrWhiteSpace($Backend)) {
  $Backend = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}

$requireSession = (-not $AllowNoSession)
if ($requireSession -and [string]::IsNullOrWhiteSpace($SessionId)) {
  throw "KAI_SESSION_ID is required for UI E2E in broad-beta mode. Provide a SA360-connected session id via -SessionId or env KAI_SESSION_ID."
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

$sa360Status = $null
if ($requireSession) {
  try {
    $statusUrl = "$Backend/api/sa360/oauth/status?session_id=$([uri]::EscapeDataString($SessionId))"
    $resp = Invoke-WebRequest -Uri $statusUrl -UseBasicParsing -Method GET
    $sa360Status = $resp.Content | ConvertFrom-Json
    if (-not $sa360Status.connected) {
      throw "SA360 is not connected for the provided session id."
    }
  } catch {
    throw "UI E2E requires an SA360-connected session. OAuth status check failed: $($_.Exception.Message)"
  }
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

$counts = [ordered]@{
  passed = $null
  failed = $null
  skipped = $null
  timed_out = $null
  total_observed = $null
  gate_ok = $null
  gate_reason = $null
}
try {
  $extractLast = {
    param([string]$text, [string]$pattern)
    $m = [regex]::Matches($text, $pattern)
    if ($m.Count -eq 0) { return $null }
    return [int]$m[$m.Count - 1].Groups[1].Value
  }

  # Tee-Object can lag a moment behind the Playwright process exit. Retry reads briefly to avoid
  # a false-negative "no_summary_counts_found" gate when the summary line hasn't flushed yet.
  $text = $null
  for ($attempt = 0; $attempt -lt 5; $attempt++) {
    $text = Get-Content $logPath -Raw
    $counts.passed = & $extractLast $text '(\d+)\s+passed\b'
    $counts.failed = & $extractLast $text '(\d+)\s+failed\b'
    $counts.skipped = & $extractLast $text '(\d+)\s+skipped\b'
    $counts.timed_out = & $extractLast $text '(\d+)\s+timed out\b'
    if ($counts.passed -ne $null -and $counts.passed -gt 0) { break }
    Start-Sleep -Seconds 1
  }
  $total = 0
  foreach ($k in @('passed', 'failed', 'skipped', 'timed_out')) {
    if ($counts[$k] -ne $null) { $total += [int]$counts[$k] }
  }
  $counts.total_observed = $total

  if ($exitCode -ne 0) {
    $counts.gate_ok = $false
    $counts.gate_reason = "playwright_exit_nonzero"
  } elseif ($requireSession) {
    # Hard gate: prevent "false green" runs where most tests are skipped due to missing session/SA360 context.
    if ($counts.passed -eq $null -or $counts.total_observed -eq 0) {
      $counts.gate_ok = $false
      $counts.gate_reason = "no_summary_counts_found"
    } elseif ($counts.passed -lt 20) {
      $counts.gate_ok = $false
      $counts.gate_reason = "too_few_tests_passed"
    } elseif (($counts.failed -ne $null -and $counts.failed -gt 0) -or ($counts.timed_out -ne $null -and $counts.timed_out -gt 0)) {
      $counts.gate_ok = $false
      $counts.gate_reason = "tests_failed_or_timed_out"
    } elseif ($counts.skipped -ne $null -and $counts.skipped -gt 5) {
      $counts.gate_ok = $false
      $counts.gate_reason = "too_many_tests_skipped"
    } else {
      $counts.gate_ok = $true
    }
  } else {
    # Minimal gate for non-session mode: require some coverage and no failures.
    if ($counts.passed -eq $null -or $counts.passed -lt 5) {
      $counts.gate_ok = $false
      $counts.gate_reason = "too_few_tests_passed_no_session_mode"
    } elseif (($counts.failed -ne $null -and $counts.failed -gt 0) -or ($counts.timed_out -ne $null -and $counts.timed_out -gt 0)) {
      $counts.gate_ok = $false
      $counts.gate_reason = "tests_failed_or_timed_out"
    } else {
      $counts.gate_ok = $true
    }
  }
} catch {
  $counts.gate_ok = $false
  $counts.gate_reason = "count_parse_failed"
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
  sa360_connected = ($sa360Status -ne $null -and $sa360Status.connected -eq $true)
  exit_code = $exitCode
  passed = ($exitCode -eq 0)
  playwright_counts = $counts
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
if (-not $counts.gate_ok) {
  $reason = $counts.gate_reason
  if ([string]::IsNullOrWhiteSpace($reason)) { $reason = "unknown" }
  throw ("UI E2E gate failed: " + $reason)
}
