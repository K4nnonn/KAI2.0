param(
  [ValidateSet("offline", "online")]
  [string]$Mode = "offline",
  [string]$Frontend = $env:FRONTEND_URL,
  [string]$Backend = $env:BACKEND_URL,
  [string]$SessionId = $env:KAI_SESSION_ID,
  [string]$QaSessionId = $env:KAI_QA_SESSION_ID,
  [switch]$LiveOauthSmoke
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunBase = Join-Path $RepoRoot "verification_runs"
New-Item -ItemType Directory -Force -Path $RunBase | Out-Null
$RunDir = Join-Path $RunBase ("beta_gate_" + $Mode + "_" + $Stamp)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Write-Json($Path, $Obj) {
  $Obj | ConvertTo-Json -Depth 8 | Set-Content -Path $Path -Encoding UTF8
}

function Redact-Text($Text) {
  if ([string]::IsNullOrWhiteSpace($Text)) { return $Text }
  $redacted = $Text
  $keys = @(
    "session_id","admin_password","code","access_token","refresh_token","client_secret",
    "sig","se","sp","sv","sr","st","skoid","sktid","skt","ske","skv","sip","si",
    "key","secret","token","password","connection_string"
  )
  foreach ($k in $keys) {
    $pattern = '((?:\\?|&|\\\\u0026)' + [regex]::Escape($k) + '=).*?(?=(?:&|\\\\u0026|\\s|"))'
    $redacted = [regex]::Replace($redacted, $pattern, '$1REDACTED')
  }
  return $redacted
}

function Redact-Args([object[]]$Args) {
  if (-not $Args) { return $Args }
  $copy = @()
  for ($i = 0; $i -lt $Args.Count; $i++) {
    $copy += $Args[$i]
    $a = [string]$Args[$i]
    if ($a -in @("-SessionId", "--session-id")) {
      if ($i + 1 -lt $Args.Count) {
        $copy += "<REDACTED>"
        $i++
      }
    }
  }
  return $copy
}

if ([string]::IsNullOrWhiteSpace($Frontend)) {
  $Frontend = "https://kai-web.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}
if ([string]::IsNullOrWhiteSpace($Backend)) {
  $Backend = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}

# Unify session id envs across scripts (UI E2E uses KAI_SESSION_ID; full_qa uses KAI_QA_SESSION_ID).
$effectiveSession = $QaSessionId
if ([string]::IsNullOrWhiteSpace($effectiveSession)) { $effectiveSession = $SessionId }

$meta = [ordered]@{
  run_id = $Stamp
  mode = $Mode
  frontend = $Frontend
  backend = $Backend
  session_id_present = (-not [string]::IsNullOrWhiteSpace($effectiveSession))
  artifacts_dir = $RunDir
  steps = @()
  ok = $false
}

if ($Mode -eq "online" -and [string]::IsNullOrWhiteSpace($effectiveSession)) {
  throw "beta_gate online mode requires an SA360-connected session id. Set env KAI_QA_SESSION_ID (preferred) or KAI_SESSION_ID."
}

$env:FRONTEND_URL = $Frontend
$env:BACKEND_URL = $Backend
$env:KAI_SESSION_ID = $effectiveSession
$env:KAI_QA_SESSION_ID = $effectiveSession

try {
  $uiScript = Join-Path $RepoRoot "scripts\\verification\\ui_e2e.ps1"
  $fullQaScript = Join-Path $RepoRoot "scripts\\verification\\full_qa.ps1"

  if (-not (Test-Path $uiScript)) { throw "Missing script: $uiScript" }
  if (-not (Test-Path $fullQaScript)) { throw "Missing script: $fullQaScript" }

  $uiArgs = @("-File", $uiScript, "-Frontend", $Frontend, "-Backend", $Backend)
  if ($Mode -eq "offline") { $uiArgs += "-AllowNoSession" }
  if ($Mode -eq "online") { $uiArgs += @("-SessionId", $effectiveSession) }
  if ($LiveOauthSmoke) { $uiArgs += "-LiveOauthSmoke" }

  $meta.steps += [ordered]@{
    step = "ui_e2e"
    started_at = (Get-Date).ToString("o")
    args = (Redact-Args $uiArgs)
    status = "running"
  }
  $uiOut = & powershell -ExecutionPolicy Bypass @uiArgs 2>&1
  $uiLog = Join-Path $RunDir "ui_e2e_output.txt"
  (Redact-Text ($uiOut | Out-String)) | Set-Content -Path $uiLog -Encoding UTF8
  $meta.steps[-1].status = "ok"
  $meta.steps[-1].finished_at = (Get-Date).ToString("o")
  $meta.steps[-1].output_log = $uiLog

  $qaArgs = @("-File", $fullQaScript, "-Backend", $Backend)
  if ($Mode -eq "offline") { $qaArgs += "-SkipSa360" } else { $qaArgs += @("-SessionId", $effectiveSession) }

  $meta.steps += [ordered]@{
    step = "full_qa"
    started_at = (Get-Date).ToString("o")
    args = (Redact-Args $qaArgs)
    status = "running"
  }
  $qaOut = & powershell -ExecutionPolicy Bypass @qaArgs 2>&1
  $qaLog = Join-Path $RunDir "full_qa_output.txt"
  (Redact-Text ($qaOut | Out-String)) | Set-Content -Path $qaLog -Encoding UTF8
  $meta.steps[-1].status = "ok"
  $meta.steps[-1].finished_at = (Get-Date).ToString("o")
  $meta.steps[-1].output_log = $qaLog

  # Attach summaries (share-safe) + enforce online lane (no "false green" when SA360 is broken).
  $uiRunDir = $null
  foreach ($line in $uiOut) {
    if ($line -match '^Run folder:\s*(.+)$') { $uiRunDir = $matches[1].Trim(); break }
  }
  if ($uiRunDir -and (Test-Path $uiRunDir)) {
    $uiSummaryPath = Join-Path $uiRunDir "summary.json"
    if (Test-Path $uiSummaryPath) {
      $meta.ui_e2e = Get-Content $uiSummaryPath -Raw | ConvertFrom-Json
    }
  }

  $qaRunDir = $null
  foreach ($line in $qaOut) {
    if ($line -match '^Full QA run folder:\s*(.+)$') { $qaRunDir = $matches[1].Trim(); break }
  }
  if ($qaRunDir -and (Test-Path $qaRunDir)) {
    $qaSummaryPath = Join-Path $qaRunDir "summary.json"
    if (Test-Path $qaSummaryPath) {
      $meta.full_qa = Get-Content $qaSummaryPath -Raw | ConvertFrom-Json
    }
  }

  if ($Mode -eq "online") {
    if (-not $meta.ui_e2e) { throw "Online gate failed: missing ui_e2e summary." }
    if (-not $meta.ui_e2e.sa360_connected) { throw "Online gate failed: ui_e2e reports SA360 not connected." }
    if (-not $meta.full_qa) { throw "Online gate failed: missing full_qa summary." }

    $hardChecks = @(
      @{ key = "health_ok"; label = "API health" },
      @{ key = "diagnostics_ok"; label = "API diagnostics" },
      @{ key = "sa360_connected"; label = "SA360 connected" },
      @{ key = "sa360_conversion_actions_ok"; label = "SA360 conversion actions" },
      @{ key = "spec_eval_status"; label = "Spec eval (runner)" },
      @{ key = "sa360_parity_ok"; label = "SA360 parity (plan vs live)" },
      @{ key = "sa360_accuracy_ok"; label = "SA360 accuracy (plan coverage)" },
      @{ key = "pmax_ok"; label = "PMax endpoint" },
      @{ key = "serp_ok"; label = "SERP endpoint" },
      @{ key = "competitor_ok"; label = "Competitor endpoint" },
      @{ key = "intel_ok"; label = "Intel endpoint" },
      @{ key = "multisheet_upload_ok"; label = "Multi-sheet upload" },
      @{ key = "audit_uploaded_data_ok"; label = "Audit via uploaded data" }
    )

    $failed = @()
    foreach ($check in $hardChecks) {
      $val = $meta.full_qa.$($check.key)
      if ($check.key -eq "spec_eval_status") {
        if ($val -ne "ok") { $failed += ($check.label + " (" + $check.key + ")") }
        continue
      }
      if ($val -ne $true) { $failed += ($check.label + " (" + $check.key + ")") }
    }
    # Spec suite must be fully green (no silent failures).
    try {
      if ($meta.full_qa.spec_eval_summary -and $meta.full_qa.spec_eval_summary.failed -gt 0) {
        $failed += ("Spec eval assertions (" + [string]$meta.full_qa.spec_eval_summary.failed + " failed)")
      }
    } catch {}
    if ($failed.Count -gt 0) {
      throw ("Online gate failed: " + ($failed -join ", "))
    }
  }

  $meta.ok = $true
} catch {
  $meta.ok = $false
  $meta.error = $_.Exception.Message
  throw
} finally {
  Write-Json (Join-Path $RunDir "summary.json") $meta
  Write-Host "beta_gate run folder: $RunDir"
}
