param(
  [string]$RunsDir = "",
  [int]$LastN = 30,
  [string]$OutDir = "",
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
if ([string]::IsNullOrWhiteSpace($RunsDir)) {
  $RunsDir = Join-Path $RepoRoot "verification_runs"
}
if (-not (Test-Path $RunsDir)) {
  throw "RunsDir not found: $RunsDir"
}

function Read-Json($Path) {
  if (-not (Test-Path $Path)) { return $null }
  try {
    return (Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
  } catch {
    try { return (Get-Content -Path $Path -Raw | ConvertFrom-Json) } catch { return $null }
  }
}

function Write-Json($Path, $Obj) {
  $Obj | ConvertTo-Json -Depth 16 | Set-Content -Path $Path -Encoding UTF8
}

function Detect-Kind($Summary) {
  if ($Summary -eq $null) { return "unknown" }
  if ($Summary.PSObject.Properties["pillars"] -and $Summary.PSObject.Properties["spec_dir"]) { return "spec_eval" }
  if ($Summary.PSObject.Properties["browser"] -and $Summary.PSObject.Properties["route_budget_sec"]) { return "ui_e2e" }
  if ($Summary.PSObject.Properties["integrity_status"] -and $Summary.PSObject.Properties["route_latency_ms"]) { return "fullqa" }
  return "unknown"
}

function Parse-Timestamp($Summary) {
  if ($Summary -eq $null) { return $null }
  foreach ($k in @("timestamp", "Timestamp")) {
    if ($Summary.PSObject.Properties[$k] -and $Summary.$k) {
      try { return [datetime]::Parse([string]$Summary.$k) } catch {}
    }
  }
  # Fallback: for fullqa/ui_e2e run_id fields are yyyymmdd_hhmmss-like.
  foreach ($k in @("run_id","RunId")) {
    if ($Summary.PSObject.Properties[$k] -and $Summary.$k) {
      $s = [string]$Summary.$k
      if ($s -match '^(\\d{8})_(\\d{6})$') {
        try { return [datetime]::ParseExact($s, "yyyyMMdd_HHmmss", $null) } catch {}
      }
    }
  }
  return $null
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $OutDir = Join-Path $RunsDir ("online_monitor_" + $Stamp)
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$records = @()
foreach ($d in (Get-ChildItem -Path $RunsDir -Directory | Sort-Object Name)) {
  $summaryPath = Join-Path $d.FullName "summary.json"
  if (-not (Test-Path $summaryPath)) { continue }
  $summary = Read-Json $summaryPath
  if (-not $summary) { continue }
  $kind = Detect-Kind $summary
  $ts = Parse-Timestamp $summary
  $tsStr = $null
  if ($ts) { $tsStr = $ts.ToString("o") }
  $records += [ordered]@{
    kind = $kind
    dir = $d.FullName
    summary_path = $summaryPath
    timestamp = $tsStr
    summary = $summary
  }
}

# Keep only the most recent N records (by timestamp when available, else by directory name).
$sorted = $records | Sort-Object { $_.timestamp } -Descending
if ($sorted.Count -gt $LastN) { $sorted = $sorted | Select-Object -First $LastN }

function Summarize-SpecEval($Rec) {
  $dir = $Rec.dir
  $resultsPath = Join-Path $dir "results.json"
  $results = Read-Json $resultsPath
  $byPillar = $Rec.summary.pillars
  $failed = @()
  if ($results) {
    $failed = @($results | Where-Object { $_.ok -ne $true } | Select-Object -First 10 id, pillar, kind, latency_ms, failures)
  }
  return [ordered]@{
    out_dir = $dir
    total = $Rec.summary.total
    passed = $Rec.summary.passed
    failed = $Rec.summary.failed
    pillars = $byPillar
    sample_failures = $failed
  }
}

function Summarize-FullQa($Rec) {
  $s = $Rec.summary
  return [ordered]@{
    run_id = $s.run_id
    backend = $s.backend
    health_ok = $s.health_ok
    diagnostics_ok = $s.diagnostics_ok
    sa360_connected = $s.sa360_connected
    route_latency_ms = $s.route_latency_ms
    route_latency_ok = $s.route_latency_ok
    route_verify_failed = $s.route_verify_failed
    default_account_saved = $s.default_account_saved
    plan_empty_ids_ok = $s.plan_empty_ids_ok
    plan_latency_ms = $s.plan_latency_ms
    plan_latency_ok = $s.plan_latency_ok
    chat_latency_ms = $s.chat_latency_ms
    chat_latency_ok = $s.chat_latency_ok
    custom_metric_relational_inferred = $s.custom_metric_relational_inferred
    custom_metric_relational_drivers_ok = $s.custom_metric_relational_drivers_ok
    audit_plan_and_run_ok = $s.audit_plan_and_run_ok
  }
}

function Summarize-UiE2e($Rec) {
  $s = $Rec.summary
  return [ordered]@{
    run_id = $s.run_id
    frontend = $s.frontend
    backend = $s.backend
    browser = $s.browser
    passed = $s.passed
    exit_code = $s.exit_code
    route_budget_sec = $s.route_budget_sec
    send_budget_sec = $s.send_budget_sec
    artifacts = $s.artifacts
  }
}

$latest = [ordered]@{}
$latest.spec_eval = $null
$latest.fullqa = $null
$latest.ui_e2e = $null

foreach ($k in @("spec_eval","fullqa","ui_e2e")) {
  $first = ($sorted | Where-Object { $_.kind -eq $k } | Select-Object -First 1)
  if ($first) {
    if ($k -eq "spec_eval") { $latest.spec_eval = Summarize-SpecEval $first }
    if ($k -eq "fullqa") { $latest.fullqa = Summarize-FullQa $first }
    if ($k -eq "ui_e2e") { $latest.ui_e2e = Summarize-UiE2e $first }
  }
}

# Drift-friendly series: only capture the small set of metrics we care about for budgets and regressions.
$series = [ordered]@{
  spec_eval = @()
  fullqa = @()
  ui_e2e = @()
}
foreach ($r in $sorted) {
  if ($r.kind -eq "spec_eval") {
    $series.spec_eval += [ordered]@{
      timestamp = $r.timestamp
      passed = $r.summary.passed
      failed = $r.summary.failed
      pillars = $r.summary.pillars
    }
  } elseif ($r.kind -eq "fullqa") {
    $s = $r.summary
    $series.fullqa += [ordered]@{
      timestamp = $r.timestamp
      route_latency_ms = $s.route_latency_ms
      route_verify_failed = $s.route_verify_failed
      plan_latency_ms = $s.plan_latency_ms
      chat_latency_ms = $s.chat_latency_ms
      sa360_connected = $s.sa360_connected
    }
  } elseif ($r.kind -eq "ui_e2e") {
    $s = $r.summary
    $series.ui_e2e += [ordered]@{
      timestamp = $r.timestamp
      passed = $s.passed
      exit_code = $s.exit_code
      browser = $s.browser
    }
  }
}

$report = [ordered]@{
  timestamp = (Get-Date).ToString("o")
  runs_dir = $RunsDir
  out_dir = $OutDir
  last_n = $LastN
  total_scanned = $records.Count
  total_in_window = $sorted.Count
  latest = $latest
  series = $series
}

$outPath = Join-Path $OutDir "report.json"
Write-Json $outPath $report
Write-Output "Online monitor folder: $OutDir"

if ($Strict) {
  $fail = $false
  if ($latest.spec_eval -and $latest.spec_eval.failed -gt 0) { $fail = $true }
  if ($latest.ui_e2e -and $latest.ui_e2e.passed -eq $false) { $fail = $true }
  if ($latest.fullqa -and $latest.fullqa.route_verify_failed -eq $true) { $fail = $true }
  if ($fail) { exit 1 }
}
