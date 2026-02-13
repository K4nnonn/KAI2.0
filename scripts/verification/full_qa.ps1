param(
  [string]$Backend = $env:KAI_BACKEND_URL,
  [string]$SessionId = $env:KAI_QA_SESSION_ID,
  [string]$LoginCustomerId = $env:KAI_SA360_LOGIN_CUSTOMER_ID,
  [string]$EnvGuiPassword = $env:KAI_ENV_GUI_PASSWORD,
  [string]$AccessPassword = $env:KAI_ACCESS_PASSWORD,
  [string]$TargetCustomerId = $env:KAI_QA_TARGET_CUSTOMER_ID,
  [int]$MaxChatLatencyMs = 20000,
  [int]$MaxPlanLatencyMs = 25000,
  [string]$CustomMetric = $env:KAI_QA_CUSTOM_METRIC,
  [switch]$SkipSa360,
  [switch]$RunSa360Audit,
  [switch]$SkipEnv
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Backend)) {
  $Backend = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
# Keep artifacts inside the Kai repo (avoid writing to unrelated folders on shared machines).
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$RunBase = Join-Path $RepoRoot "verification_runs"
New-Item -ItemType Directory -Force -Path $RunBase | Out-Null
$RunDir = Join-Path $RunBase ("fullqa_" + $Stamp)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

# Deterministic windows: avoid flaky metrics when very recent days drift. Use a stable 7-day window
# that ends 2 days ago (complete days), plus a longer stable window for driver-style prompts.
$stableEnd = (Get-Date).Date.AddDays(-2)
$stableStart7 = $stableEnd.AddDays(-6)
$stableStart30 = $stableEnd.AddDays(-29)
$STABLE_LAST_7_DAYS = ("{0:yyyy-MM-dd},{1:yyyy-MM-dd}" -f $stableStart7, $stableEnd)
$STABLE_LAST_30_DAYS = ("{0:yyyy-MM-dd},{1:yyyy-MM-dd}" -f $stableStart30, $stableEnd)

function Write-Json($Path, $Obj) {
  $json = $Obj | ConvertTo-Json -Depth 12
  # Make verification artifacts safe to share: redact secrets even if they appear inside response bodies.
  $json = Redact-JsonText $json
  $json | Set-Content -Path $Path -Encoding UTF8
}

function Redact-Url($Url) {
  if ([string]::IsNullOrWhiteSpace($Url)) { return $Url }
  $redacted = $Url
  # Redact common sensitive query params (keep artifacts safe to share inside the team).
  $keys = @(
    "session_id",
    "admin_password",
    "code",
    "access_token",
    "refresh_token",
    "client_secret",
    # Azure SAS / signed URLs (common in blob download links)
    "sig",
    "se",
    "sp",
    "sv",
    "sr",
    "st",
    "skoid",
    "sktid",
    "skt",
    "ske",
    "skv",
    "sip",
    "si"
  )
  foreach ($k in $keys) {
    $pattern = "([?&]" + [regex]::Escape($k) + "=)[^&]+"
    $redacted = [regex]::Replace($redacted, $pattern, "`$1REDACTED")
  }
  return $redacted
}

function Redact-JsonText($Json) {
  if ([string]::IsNullOrWhiteSpace($Json)) { return $Json }
  $redacted = $Json

  # 1) Redact sensitive query params inside any URL-like strings in JSON (handles '&' and '\u0026').
  $queryKeys = @(
    "session_id",
    "admin_password",
    "code",
    "access_token",
    "refresh_token",
    "client_secret",
    "sig","se","sp","sv","sr","st","skoid","sktid","skt","ske","skv","sip","si"
  )
  foreach ($k in $queryKeys) {
    $pattern = '((?:\\?|&|\\\\u0026)' + [regex]::Escape($k) + '=).*?(?=(?:&|\\\\u0026|"))'
    $redacted = [regex]::Replace($redacted, $pattern, '$1REDACTED')
  }

  # 2) Redact direct secret fields if they appear as JSON keys.
  $directKeys = @(
    "session_id",
    "admin_password",
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "accountKey",
    "connectionString"
  )
  foreach ($k in $directKeys) {
    # Match JSON key/value pairs (whitespace is real whitespace, not the literal string "\s").
    $pattern = '("' + [regex]::Escape($k) + '"\s*:\s*")[^"]+(")'
    $redacted = [regex]::Replace($redacted, $pattern, '$1REDACTED$2')
  }

  return $redacted
}

function Safe-Request($Method, $Url, $Body = $null) {
  $result = [ordered]@{
    url = (Redact-Url $Url)
    method = $Method
    status = $null
    ok = $false
    error = $null
    body = $null
  }
  try {
    if ($Body -ne $null) {
      $resp = Invoke-WebRequest -Method $Method -Uri $Url -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 8) -UseBasicParsing
    } else {
      $resp = Invoke-WebRequest -Method $Method -Uri $Url -UseBasicParsing
    }
    $result.status = $resp.StatusCode
    $result.ok = ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300)
    if ($resp.Content) {
      try {
        $result.body = $resp.Content | ConvertFrom-Json
      } catch {
        $result.body = $resp.Content
      }
    }
  } catch {
    $err = $_
    if ($err.Exception.Response) {
      $result.status = [int]$err.Exception.Response.StatusCode
      try {
        $reader = New-Object System.IO.StreamReader($err.Exception.Response.GetResponseStream())
        $content = $reader.ReadToEnd()
        try {
          $result.body = $content | ConvertFrom-Json
        } catch {
          $result.body = $content
        }
      } catch {}
    }
    $result.error = $err.Exception.Message
  }
  return $result
}

function Parse-Json($Text) {
  if ($Text -eq $null) { return $null }
  try { return ($Text | ConvertFrom-Json) } catch { return $null }
}

# --- LLM usage snapshot (cost/behavior evidence) ---
$llmUsageBefore = Safe-Request "GET" "$Backend/api/diagnostics/llm-usage"
Write-Json (Join-Path $RunDir "00_llm_usage_before.json") $llmUsageBefore

# --- Run core integrity script ---
$integrity = [ordered]@{ status = "skipped"; reason = "missing_script" }
$integrityScript = Join-Path $PSScriptRoot "verify_system_integrity.ps1"
if (Test-Path $integrityScript) {
  $env:KAI_BACKEND_URL = $Backend
  $output = & powershell -ExecutionPolicy Bypass -File $integrityScript 2>&1
  $runFolder = $null
  foreach ($line in $output) {
    if ($line -match "Run folder:\s*(.+)$") {
      $runFolder = $matches[1].Trim()
      break
    }
  }
  if ($runFolder -and (Test-Path $runFolder)) {
    $summaryPath = Join-Path $runFolder "summary.json"
    $summary = $null
    if (Test-Path $summaryPath) {
      $summary = Get-Content $summaryPath | ConvertFrom-Json
    }
    $integrity = [ordered]@{
      status = "ok"
      run_folder = $runFolder
      summary = $summary
    }
  } else {
    $integrity = [ordered]@{
      status = "error"
      output = $output
    }
  }
}
Write-Json (Join-Path $RunDir "00_integrity.json") $integrity

# --- Extra QA checks ---
$health = Safe-Request "GET" "$Backend/api/health"
$diag = Safe-Request "GET" "$Backend/api/diagnostics/health"
Write-Json (Join-Path $RunDir "01_health.json") $health
Write-Json (Join-Path $RunDir "02_diagnostics_health.json") $diag

# Env checks
$envChecks = [ordered]@{
  skipped = $false
  reason = $null
  no_auth = $null
  with_password = $null
}
if ($SkipEnv -or [string]::IsNullOrWhiteSpace($EnvGuiPassword)) {
  $envChecks.skipped = $true
  $envChecks.reason = "missing_env_gui_password"
} else {
  $envChecks.no_auth = Safe-Request "GET" "$Backend/api/settings/env"
  $envChecks.with_password = Safe-Request "GET" "$Backend/api/settings/env?admin_password=$([uri]::EscapeDataString($EnvGuiPassword))"
}
Write-Json (Join-Path $RunDir "03_env_checks.json") $envChecks

# Auth verify (UI password gate depends on this)
$authChecks = [ordered]@{
  wrong_rejected = $null
  wrong_rejected_ok = $null
  correct_ok = $null
  correct_ok_ok = $null
  correct_skipped = $false
  reason = $null
}
$wrongReq = @{ password = ("invalid-" + [guid]::NewGuid().ToString("n")) }
$wrongResp = Safe-Request "POST" "$Backend/api/auth/verify" $wrongReq
$authChecks.wrong_rejected = $wrongResp
$authChecks.wrong_rejected_ok = ($wrongResp.status -eq 401)
if ([string]::IsNullOrWhiteSpace($AccessPassword)) {
  $authChecks.correct_skipped = $true
  $authChecks.reason = "missing_access_password"
} else {
  $okReq = @{ password = $AccessPassword }
  $okResp = Safe-Request "POST" "$Backend/api/auth/verify" $okReq
  $authChecks.correct_ok = $okResp
  $authChecks.correct_ok_ok = ($okResp.ok -eq $true -and $okResp.body -and $okResp.body.authenticated -eq $true)
}
Write-Json (Join-Path $RunDir "03a_auth_verify.json") $authChecks

# Creative
$creativeReq = @{
  business_name = "Kai QA"
  url = "https://example.com"
  keywords = @("paid search", "sa360")
  usps = @("fast setup", "advisor quality")
}
$creative = Safe-Request "POST" "$Backend/api/creative/generate" $creativeReq
Write-Json (Join-Path $RunDir "04_creative.json") $creative

# Audit missing data (should return 400 with clear reason)
$auditReq = @{
  business_unit = "Brand"
  account_name = "QA Account"
  use_mock_data = $true
  async_mode = $false
}
$audit = Safe-Request "POST" "$Backend/api/audit/generate" $auditReq
Write-Json (Join-Path $RunDir "05_audit_missing_data.json") $audit

# Advisor diagnostics (no numeric evidence -> expect fail + missing_numeric_evidence)
$advisorReq = @{
  text = "No performance data available."
  use_case = "performance"
}
$advisor = Safe-Request "POST" "$Backend/api/diagnostics/advisor" $advisorReq
Write-Json (Join-Path $RunDir "06_advisor_diagnostics.json") $advisor

# SA360 validation (requires session ID)
$sa360 = [ordered]@{
  skipped = $false
  reason = $null
  oauth_status = $null
  accounts = $null
  fetch = $null
  fetch_and_audit = $null
  job = $null
}
if ($SkipSa360) {
  $sa360.skipped = $true
  $sa360.reason = "skip_flag"
} elseif ([string]::IsNullOrWhiteSpace($SessionId)) {
  $sa360.skipped = $true
  $sa360.reason = "missing_session_id"
} else {
  $statusResp = Safe-Request "GET" "$Backend/api/sa360/oauth/status?session_id=$([uri]::EscapeDataString($SessionId))"
  $sa360.oauth_status = $statusResp
  # Prefer a stable, user-saved default account for QA runs (reduces flakiness across large MCCs).
  # This keeps broad-beta regression gates grounded in the same account a user would actually operate on.
  try {
    if (-not $TargetCustomerId -and $statusResp.ok -and $statusResp.body -and $statusResp.body.default_customer_id) {
      $TargetCustomerId = [string]$statusResp.body.default_customer_id
    }
  } catch {}
  $connected = $false
  if ($statusResp.ok -and $statusResp.body -and $statusResp.body.connected -eq $true) {
    $connected = $true
  }
  if (-not $connected) {
    $sa360.reason = "not_connected"
  } else {
    $accountsResp = Safe-Request "GET" "$Backend/api/sa360/accounts?session_id=$([uri]::EscapeDataString($SessionId))"
    $sa360.accounts = $accountsResp
    $accounts = @()
    if ($accountsResp.ok -and $accountsResp.body) {
      $accounts = $accountsResp.body
    }
    if (-not $accounts -or $accounts.Count -eq 0) {
      $sa360.reason = "no_accounts"
    } else {
      if ($TargetCustomerId) {
        $target = ($accounts | Where-Object { $_.customer_id -eq $TargetCustomerId } | Select-Object -First 1)
      }
      if (-not $target) {
        $target = ($accounts | Where-Object { $_.manager -ne $true } | Select-Object -First 1)
        if (-not $target) { $target = $accounts | Select-Object -First 1 }
      }
      $targetId = $target.customer_id

      # Conversion-action catalog (metric discovery + QA grounding). Trim payload to keep run artifacts small.
      $convResp = Safe-Request "GET" "$Backend/api/sa360/conversion-actions?session_id=$([uri]::EscapeDataString($SessionId))&customer_id=$($targetId)&date_range=LAST_30_DAYS"
      try {
        if ($convResp.ok -and $convResp.body -and $convResp.body.actions) {
          $count = ($convResp.body.actions | Measure-Object).Count
          $convResp.body | Add-Member -NotePropertyName actions_count -NotePropertyValue $count -Force

          # Pick a high-volume conversion action that is excluded from "Conversions" (conversions=0 but all_conversions>0).
          # This is a key beta risk: users ask about these columns (e.g., Store visits) and expect analysis + drivers.
          $excluded = @()
          try {
            $excluded = $convResp.body.actions | Where-Object { $_.status -eq "ENABLED" -and $_.all_conversions -gt 0 -and $_.conversions -eq 0 } | Sort-Object -Property all_conversions -Descending
          } catch {}
          $candidate = $null
          if ($excluded -and $excluded.Count -gt 0) {
            # Prefer a shorter name so we don't rely on special casing in prompt parsing.
            $candidate = ($excluded | Where-Object { $_.name -and $_.name.Length -le 80 } | Select-Object -First 1)
            if (-not $candidate) { $candidate = $excluded | Select-Object -First 1 }
          }
          if ($candidate) {
            $convResp.body | Add-Member -NotePropertyName excluded_action_candidate -NotePropertyValue ([ordered]@{
              name = $candidate.name
              metric_key = $candidate.metric_key
              category = $candidate.category
              conversions = $candidate.conversions
              all_conversions = $candidate.all_conversions
            }) -Force
          }

          if ($count -gt 40) {
            $convResp.body.actions = $convResp.body.actions | Select-Object -First 40
          }
        }
      } catch {}
      $sa360.conversion_actions = $convResp

      $fetchReq = @{
        customer_ids = @($targetId)
        date_range = "LAST_7_DAYS"
        session_id = $SessionId
        dry_run = $false
        async_mode = $false
      }
      if ($LoginCustomerId) {
        $fetchReq.login_customer_id = $LoginCustomerId
      }
      if ($RunSa360Audit) {
        $fetchReq.async_mode = $true
        $auditResp = Safe-Request "POST" "$Backend/api/integrations/sa360/fetch-and-audit" $fetchReq
        $sa360.fetch_and_audit = $auditResp
        if ($auditResp.ok -and $auditResp.body -and $auditResp.body.status -eq "queued" -and $auditResp.body.job_id) {
          $jobId = $auditResp.body.job_id
          $job = [ordered]@{
            job_id = $jobId
            status = $null
            result = $null
            attempts = 0
            timed_out = $false
          }
          $deadline = (Get-Date).AddMinutes(12)
          while ((Get-Date) -lt $deadline) {
            $job.attempts++
            $jobStatus = Safe-Request "GET" "$Backend/api/jobs/$jobId"
            $job.status = $jobStatus
            $state = $null
            if ($jobStatus.ok -and $jobStatus.body -and $jobStatus.body.job) {
              $state = $jobStatus.body.job.status
            }
            if ($state -eq "succeeded" -or $state -eq "failed") {
              break
            }
            Start-Sleep -Seconds 10
          }
          if (-not ($job.status.ok -and $job.status.body -and $job.status.body.job)) {
            $job.timed_out = $true
          } else {
            $state = $job.status.body.job.status
            if ($state -ne "succeeded" -and $state -ne "failed") {
              $job.timed_out = $true
            }
          }
          if (-not $job.timed_out) {
            $jobResult = Safe-Request "GET" "$Backend/api/jobs/$jobId/result"
            $job.result = $jobResult
          }
          $sa360.job = $job
        }
      } else {
        $fetchResp = Safe-Request "POST" "$Backend/api/integrations/sa360/fetch" $fetchReq
        $sa360.fetch = $fetchResp
      }
    }
  }
}
Write-Json (Join-Path $RunDir "07_sa360.json") $sa360

# --- Router + performance QA (latency + debug notes) ---
$routerPerf = [ordered]@{
  skipped = $false
  reason = $null
  route_latency_ms = $null
  route_samples_ms = @()
  route_p50_ms = $null
  route_p95_ms = $null
  route_min_ms = $null
  route_max_ms = $null
  # Router is Azure-first in broad beta; p95 under ~2.5s is a realistic SLA for reliability.
  route_latency_budget_ms = 2500
  route_latency_ok = $null
  route_notes = $null
  route_verify_failed = $null
  latency_ms = $null
  latency_budget_ms = $MaxPlanLatencyMs
  router_notes = $null
  router_verify_failed = $null
  perf_status = $null
  custom_metric = $CustomMetric
  custom_metric_present = $null
  custom_metric_key = $null
  custom_metric_key_ok = $null
  custom_metric_relational = $null
  default_account_saved = $null
  plan_empty_ids_ok = $null
  plan_empty_ids_response = $null
}
if ($sa360.skipped -or -not $sa360.accounts -or $sa360.reason) {
  $routerPerf.skipped = $true
  $routerPerf.reason = "sa360_not_ready"
} else {
  $accounts = $sa360.accounts.body
  if ($TargetCustomerId) {
    $target = ($accounts | Where-Object { $_.customer_id -eq $TargetCustomerId } | Select-Object -First 1)
  }
  if (-not $target) {
    $target = ($accounts | Where-Object { $_.manager -ne $true } | Select-Object -First 1)
    if (-not $target) { $target = $accounts | Select-Object -First 1 }
  }
  $targetId = $target.customer_id

  # Save a default account so planner can run even if the user doesn't paste IDs (beta-critical UX).
  $defaultReq = @{
    session_id = $SessionId
    customer_id = $targetId
    account_name = $target.name
  }
  $defaultSave = Safe-Request "POST" "$Backend/api/sa360/default-account" $defaultReq
  $routerPerf.default_account_saved = $defaultSave.ok

  $emptyIdsReq = @{
    message = "How did performance look recently?"
    customer_ids = @()
    session_id = $SessionId
    default_date_range = $STABLE_LAST_7_DAYS
  }
  $emptyIdsResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $emptyIdsReq
  $routerPerf.plan_empty_ids_response = $emptyIdsResp
  try {
    $planned = $emptyIdsResp.body.plan.customer_ids
    $routerPerf.plan_empty_ids_ok = (
      $emptyIdsResp.ok -and
      $emptyIdsResp.body.executed -eq $true -and
      $planned -and
      ($planned | Where-Object { $_ -eq $targetId } | Measure-Object).Count -ge 1
    )
  } catch {
    $routerPerf.plan_empty_ids_ok = $false
  }

  # Router should be fast and must not leak local verification failures in a broad beta.
  $routeReq = @{
    message = "How did performance look recently?"
    customer_ids = @($targetId)
    session_id = $SessionId
    default_date_range = $STABLE_LAST_7_DAYS
  }
  $routeSamples = @()
  $routeNotes = @()
  $routeVerifyFailedAny = $false
  $routeSampleCount = 10
  for ($i = 0; $i -lt $routeSampleCount; $i++) {
    $routeTimer = Measure-Command {
      $routeResp = Safe-Request "POST" "$Backend/api/chat/route" $routeReq
    }
    $ms = [math]::Round($routeTimer.TotalMilliseconds, 2)
    $routeSamples += $ms
    if ($routeResp.body -and $routeResp.body.notes) {
      $note = [string]$routeResp.body.notes
      if ($note) { $routeNotes += $note }
      if ($note -like "*router_verify_failed*") { $routeVerifyFailedAny = $true }
    }
    Start-Sleep -Milliseconds 100
  }
  $routerPerf.route_samples_ms = $routeSamples
  $sorted = $routeSamples | Sort-Object
  if ($sorted.Count -gt 0) {
    $routerPerf.route_min_ms = $sorted[0]
    $routerPerf.route_max_ms = $sorted[-1]
    $routerPerf.route_p50_ms = $sorted[[math]::Floor(0.50 * ($sorted.Count - 1))]
    $routerPerf.route_p95_ms = $sorted[[math]::Floor(0.95 * ($sorted.Count - 1))]
    # Back-compat: treat route_latency_ms as p95 (budget is about reliability, not best-case).
    $routerPerf.route_latency_ms = $routerPerf.route_p95_ms
  }
  $routerPerf.route_notes = (($routeNotes | Select-Object -Unique) -join "; ")
  $routerPerf.route_verify_failed = $routeVerifyFailedAny
  $routerPerf.route_latency_ok = ($routerPerf.route_latency_ms -ne $null -and $routerPerf.route_latency_ms -le $routerPerf.route_latency_budget_ms)

  $planReq = @{
    message = "How did performance look recently?"
    customer_ids = @($targetId)
    session_id = $SessionId
    default_date_range = $STABLE_LAST_7_DAYS
  }
  $timer = Measure-Command {
    $planResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $planReq
  }
  $routerPerf.latency_ms = [math]::Round($timer.TotalMilliseconds, 2)
  $routerPerf.perf_status = $planResp
  if ($planResp.body -and $planResp.body.notes) {
    $routerPerf.router_notes = $planResp.body.notes
    $routerPerf.router_verify_failed = ($planResp.body.notes -like "*router_verify_failed*")
  }
  if ($CustomMetric) {
    $customReq = @{
      message = "Show me $CustomMetric for the selected period."
      customer_ids = @($targetId)
      session_id = $SessionId
      default_date_range = $STABLE_LAST_7_DAYS
    }
    $customResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $customReq
    $routerPerf.custom_metric_present = $false
    if ($customResp.body -and $customResp.body.result -and $customResp.body.result.current) {
      $current = $customResp.body.result.current
      foreach ($k in $current.PSObject.Properties.Name) {
        if ($k -like "custom:*") { $routerPerf.custom_metric_present = $true }
      }
    }
    try {
      if ($customResp.ok -and $customResp.body -and $customResp.body.result -and $customResp.body.result.data_quality) {
        $routerPerf.custom_metric_key = $customResp.body.result.data_quality.custom_metric_key
        $routerPerf.custom_metric_key_ok = ($routerPerf.custom_metric_key -ne $null -and [string]$routerPerf.custom_metric_key -like "custom:*")
      }
    } catch {}
    $routerPerf.custom_metric_response = $customResp
  }

  # --- Custom metric quality gates (beta-critical) ---
  # 1) If a user pastes an explicit snake_case token (e.g., FR_Intent_Clicks) and it doesn't exist in SA360,
  #    the system must NOT silently remap it. It should block + suggest alternatives.
  $explicitReq = @{
    message = "Why did FR_Intent_Clicks change week over week? Which campaigns drove it?"
    customer_ids = @($targetId)
    session_id = $SessionId
    default_date_range = $STABLE_LAST_30_DAYS
  }
  $explicitTimer = Measure-Command {
    $explicitResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $explicitReq
  }
  $explicitLatency = [math]::Round($explicitTimer.TotalMilliseconds, 2)
  $explicitBlocked = $false
  $explicitSuggestions = @()
  try {
    $explicitBlocked = (
      $explicitResp.ok -and
      $explicitResp.body -and
      $explicitResp.body.executed -eq $false -and
      $explicitResp.body.error -eq "custom_metric_not_found"
    )
    $explicitSuggestions = @($explicitResp.body.analysis.custom_metric.suggestions) | Where-Object { $_ }
  } catch {}
  $routerPerf.custom_metric_explicit_token = [ordered]@{
    latency_ms = $explicitLatency
    blocked = $explicitBlocked
    suggestions_count = ($explicitSuggestions | Measure-Object).Count
    suggestions = $explicitSuggestions
    response = $explicitResp
  }

  # 2) For a real conversion action name (e.g., Store visits), the system should infer + run drivers.
  $storeReq = @{
    message = "Why did Store visits change week over week? Which campaigns drove it?"
    customer_ids = @($targetId)
    session_id = $SessionId
    default_date_range = $STABLE_LAST_30_DAYS
  }
  $storeTimer = Measure-Command {
    $storeResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $storeReq
  }
  $storeLatency = [math]::Round($storeTimer.TotalMilliseconds, 2)
  $storeInferred = $false
  $storeKey = $null
  $storeDriversOk = $false
  try {
    if ($storeResp.ok -and $storeResp.body -and $storeResp.body.result -and $storeResp.body.result.data_quality) {
      $storeInferred = ($storeResp.body.result.data_quality.custom_metric_inferred -eq $true)
      $storeKey = $storeResp.body.result.data_quality.custom_metric_key
    }
    if ($storeResp.ok -and $storeResp.body -and $storeResp.body.analysis -and $storeResp.body.analysis.drivers) {
      $campCount = ($storeResp.body.analysis.drivers.campaign | Measure-Object).Count
      $devCount = ($storeResp.body.analysis.drivers.device | Measure-Object).Count
      $storeDriversOk = ($campCount -ge 1 -or $devCount -ge 1)
    }
  } catch {}
  $routerPerf.custom_metric_relational = [ordered]@{
    latency_ms = $storeLatency
    inferred = $storeInferred
    metric_key = $storeKey
    drivers_ok = $storeDriversOk
    response = $storeResp
  }

  # Generic relational prompt should NOT trigger custom-metric suggestions (regression guardrail).
  $genericReq = @{
    message = "Why is performance down week over week?"
    customer_ids = @($targetId)
    session_id = $SessionId
    default_date_range = $STABLE_LAST_30_DAYS
  }
  $genericTimer = Measure-Command {
    $genericResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $genericReq
  }
  $genericLatency = [math]::Round($genericTimer.TotalMilliseconds, 2)
  $genericOk = $false
  $genericHadSuggestions = $null
  $genericSummaryHasNoClose = $null
  try {
    $dq = $genericResp.body.result.data_quality
    $suggestions = $dq.custom_metric_suggestions
    $genericHadSuggestions = ($suggestions -ne $null -and ($suggestions | Measure-Object).Count -gt 0)
    $sum = [string]$genericResp.body.summary
    $genericSummaryHasNoClose = ($sum -like "*No close metric match found*")
    $genericOk = ($genericResp.ok -and -not $genericHadSuggestions -and -not $genericSummaryHasNoClose)
  } catch {}
  $routerPerf.generic_relational_no_custom = [ordered]@{
    latency_ms = $genericLatency
    ok = $genericOk
    had_suggestions = $genericHadSuggestions
    summary_has_no_close = $genericSummaryHasNoClose
    response = $genericResp
  }

  # --- Audit intent regression (sa360_fetch_and_audit signature mismatch) ---
  # This specifically guards against the runtime error observed in beta:
  # "sa360_fetch_and_audit() missing 1 required positional argument: 'request'".
  $auditPlanReq = @{
    message = "Run a PPC audit for this account."
    customer_ids = @($targetId)
    session_id = $SessionId
    intent_hint = "audit"
    async_mode = $false
    generate_report = $false
    default_date_range = $STABLE_LAST_7_DAYS
  }
  $auditTimer = Measure-Command {
    $auditPlanResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $auditPlanReq
  }
  $auditLatency = [math]::Round($auditTimer.TotalMilliseconds, 2)
  $auditHadSignatureError = $false
  try {
    $sigText = ""
    if ($auditPlanResp.error) { $sigText += " " + [string]$auditPlanResp.error }
    if ($auditPlanResp.body -and $auditPlanResp.body.error) { $sigText += " " + [string]$auditPlanResp.body.error }
    if ($auditPlanResp.body -and $auditPlanResp.body.summary) { $sigText += " " + [string]$auditPlanResp.body.summary }
    if ($sigText -like "*missing 1 required positional argument*") { $auditHadSignatureError = $true }
  } catch {}
  $routerPerf.audit_plan_and_run = [ordered]@{
    latency_ms = $auditLatency
    ok = $auditPlanResp.ok
    status = $auditPlanResp.status
    signature_error = $auditHadSignatureError
    response = $auditPlanResp
  }

  # Custom metric inference must work for at least one high-volume conversion action that is excluded from "Conversions"
  # (conversions=0 but all_conversions>0). Pick it dynamically from the account catalog each run.
  $exCandidate = $null
  try {
    if ($sa360.conversion_actions -and $sa360.conversion_actions.body -and $sa360.conversion_actions.body.excluded_action_candidate) {
      $exCandidate = $sa360.conversion_actions.body.excluded_action_candidate
    }
  } catch {}
  if ($exCandidate -and $exCandidate.name) {
    $exName = [string]$exCandidate.name
    $exReq = @{
      message = "Why did $exName change week over week? Which campaigns drove it?"
      customer_ids = @($targetId)
      session_id = $SessionId
      default_date_range = $STABLE_LAST_30_DAYS
    }
    $exTimer = Measure-Command {
      $exResp = Safe-Request "POST" "$Backend/api/chat/plan-and-run" $exReq
    }
    $exLatency = [math]::Round($exTimer.TotalMilliseconds, 2)
    $exInferred = $false
    $exDriversOk = $false
    $exUsesAllConv = $false
    $exMetricCol = $null
    $exMetricFocus = $null
    $exKey = $null
    try {
      if ($exResp.ok -and $exResp.body -and $exResp.body.result -and $exResp.body.result.data_quality) {
        $exInferred = ($exResp.body.result.data_quality.custom_metric_inferred -eq $true)
        $exMetricCol = $exResp.body.result.data_quality.custom_metric_metric_col
        $exKey = $exResp.body.result.data_quality.custom_metric_key
        $exUsesAllConv = ($exMetricCol -eq "metrics.all_conversions")
      }
      if ($exResp.ok -and $exResp.body -and $exResp.body.analysis) {
        $exMetricFocus = $exResp.body.analysis.metric_focus
      }
      if ($exResp.ok -and $exResp.body -and $exResp.body.analysis -and $exResp.body.analysis.drivers) {
        $campCount = ($exResp.body.analysis.drivers.campaign | Measure-Object).Count
        $devCount = ($exResp.body.analysis.drivers.device | Measure-Object).Count
        $exDriversOk = ($campCount -ge 1 -or $devCount -ge 1)
      }
    } catch {}
    $routerPerf.custom_metric_excluded_action = [ordered]@{
      name = $exName
      expected_metric_key = $exCandidate.metric_key
      latency_ms = $exLatency
      inferred = $exInferred
      metric_key = $exKey
      metric_focus = $exMetricFocus
      metric_col = $exMetricCol
      drivers_ok = $exDriversOk
      uses_all_conversions = $exUsesAllConv
      response = $exResp
    }
  } else {
    $routerPerf.custom_metric_excluded_action = [ordered]@{
      skipped = $true
      reason = "no_excluded_action_candidate"
      candidate = $exCandidate
    }
  }
}
Write-Json (Join-Path $RunDir "08_router_perf.json") $routerPerf

# --- Chat latency budget check ---
$chatLatency = [ordered]@{
  latency_ms = $null
  latency_budget_ms = $MaxChatLatencyMs
  ok = $null
}
$chatReq = @{ message = "What can you do?"; session_id = $SessionId }
$timerChat = Measure-Command {
  $chatResp = Safe-Request "POST" "$Backend/api/chat/send" $chatReq
}
$chatLatency.latency_ms = [math]::Round($timerChat.TotalMilliseconds, 2)
$chatLatency.ok = ($chatLatency.latency_ms -le $MaxChatLatencyMs)
$chatLatency.response = $chatResp
Write-Json (Join-Path $RunDir "09_chat_latency.json") $chatLatency

# --- SA360 not-connected UX check (API) ---
$sa360No = Safe-Request "GET" "$Backend/api/sa360/oauth/status?session_id=qa_missing_session"
Write-Json (Join-Path $RunDir "10_sa360_not_connected.json") $sa360No

# --- Module smoke tests (system-wide) ---
# PMax
$pmaxReq = @{
  placements = @(
    @{ placement = "example.com"; cost = 10; ad_network_type = "YOUTUBE" }
  )
  spend = 10
}
$pmax = Safe-Request "POST" "$Backend/api/pmax/analyze" $pmaxReq
Write-Json (Join-Path $RunDir "11_pmax.json") $pmax

# SERP
$serpReq = @{ urls = @("https://example.com") }
$serp = Safe-Request "POST" "$Backend/api/serp/check" $serpReq
Write-Json (Join-Path $RunDir "12_serp.json") $serp

# Competitor signal
$compReq = @{
  competitor_domain = "homedepot.com"
  impression_share_current = 0.12
  impression_share_previous = 0.09
}
$comp = Safe-Request "POST" "$Backend/api/serp/competitor-signal" $compReq
Write-Json (Join-Path $RunDir "13_competitor.json") $comp

# Agentic Intel (minimal payload)
$intelReq = @{
  query = "Why is performance down?"
  pmax = @()
  creative = @()
  market = @()
  brand_terms = @("brand")
}
$intel = Safe-Request "POST" "$Backend/api/intel/diagnose" $intelReq
Write-Json (Join-Path $RunDir "14_intel.json") $intel

# --- Spec-driven eval suite (four pillars: intent/purpose/quality/functionality) ---
$specEval = [ordered]@{ status = "skipped"; reason = "missing_script"; out_dir = $null; summary = $null; output = $null }
$specScript = Join-Path $PSScriptRoot "run_specs.ps1"
if (Test-Path $specScript) {
  if ([string]::IsNullOrWhiteSpace($SessionId)) {
    $specEval = [ordered]@{
      status = "skipped"
      reason = "missing_session_id"
      out_dir = $null
      summary = $null
      output = $null
    }
  } else {
  try {
    $specOutDir = Join-Path $RunDir "spec_eval"
    New-Item -ItemType Directory -Force -Path $specOutDir | Out-Null

    $specTarget = $TargetCustomerId
    try {
      if ($targetId) { $specTarget = [string]$targetId }
    } catch {}

    $specOutput = & powershell -ExecutionPolicy Bypass -File $specScript -Backend $Backend -SessionId $SessionId -TargetCustomerId $specTarget -OutDir $specOutDir 2>&1
    $specSummaryPath = Join-Path $specOutDir "summary.json"
    $specSummary = $null
    if (Test-Path $specSummaryPath) {
      $specSummary = Get-Content $specSummaryPath | ConvertFrom-Json
    }
    $specEval = [ordered]@{
      status = "ok"
      out_dir = $specOutDir
      summary = $specSummary
      output = $specOutput
    }
  } catch {
    $specEval = [ordered]@{
      status = "error"
      error = $_.Exception.Message
    }
  }
  }
}
Write-Json (Join-Path $RunDir "15_spec_eval.json") $specEval

# --- Multi-sheet upload + manifest check (XLSX tabs should be discoverable) ---
$multisheet = [ordered]@{ status = "skipped"; reason = "missing_script"; out_dir = $null; summary = $null; output = $null }
$multisheetScript = Join-Path $PSScriptRoot "verify_multisheet_upload.ps1"
if (Test-Path $multisheetScript) {
  try {
    $msOutDir = Join-Path $RunDir "multisheet_upload"
    New-Item -ItemType Directory -Force -Path $msOutDir | Out-Null
    $msOutput = & powershell -ExecutionPolicy Bypass -File $multisheetScript -Backend $Backend -OutDir $msOutDir 2>&1
    $msSummaryPath = Join-Path $msOutDir "summary.json"
    $msSummary = $null
    if (Test-Path $msSummaryPath) {
      $msSummary = Get-Content $msSummaryPath | ConvertFrom-Json
    }
    $multisheet = [ordered]@{
      status = "ok"
      out_dir = $msOutDir
      summary = $msSummary
      output = $msOutput
    }
  } catch {
    $multisheet = [ordered]@{
      status = "error"
      error = $_.Exception.Message
    }
  }
}
Write-Json (Join-Path $RunDir "16_multisheet_upload.json") $multisheet

# --- Audit success path via uploaded reports (XLSX multi-sheet) ---
# This ensures the end-to-end "upload -> audit generate" workflow works (not just manifest discovery).
$auditUpload = [ordered]@{
  status = "skipped"
  reason = $null
  queued = $null
  job = $null
  result = $null
  ok = $false
}
if ($multisheet.status -ne "ok" -or -not $multisheet.summary -or -not $multisheet.summary.ok) {
  $auditUpload.reason = "multisheet_upload_not_ready"
} else {
  try {
    $auditUploadReq = @{
      business_unit = "QA MultiSheet"
      account_name = "QA MultiSheet"
      use_mock_data = $false
      async_mode = $true
      data_prefix = "qa"
    }

    # Prefer async mode to avoid long-running request timeouts; we still require the job to complete successfully.
    $queued = Safe-Request "POST" "$Backend/api/audit/generate" $auditUploadReq
    $auditUpload.queued = $queued
    if (-not ($queued.ok -and $queued.body -and $queued.body.status -eq "queued" -and $queued.body.job_id)) {
      throw "Expected queued audit job response."
    }

    $jobId = $queued.body.job_id
    $job = [ordered]@{
      job_id = $jobId
      status = $null
      result = $null
      attempts = 0
      timed_out = $false
    }
    $deadline = (Get-Date).AddMinutes(12)
    while ((Get-Date) -lt $deadline) {
      $job.attempts++
      $jobStatus = Safe-Request "GET" "$Backend/api/jobs/$jobId"
      $job.status = $jobStatus
      $state = $null
      if ($jobStatus.ok -and $jobStatus.body -and $jobStatus.body.job) {
        $state = $jobStatus.body.job.status
      }
      if ($state -eq "succeeded" -or $state -eq "failed") {
        break
      }
      Start-Sleep -Seconds 5
    }
    if (-not ($job.status.ok -and $job.status.body -and $job.status.body.job)) {
      $job.timed_out = $true
    } else {
      $state = $job.status.body.job.status
      if ($state -ne "succeeded" -and $state -ne "failed") {
        $job.timed_out = $true
      }
    }
    if (-not $job.timed_out) {
      $jobResult = Safe-Request "GET" "$Backend/api/jobs/$jobId/result"
      $job.result = $jobResult
    }
    $auditUpload.job = $job

    $auditUploadOk = $false
    try {
      $auditUploadOk = (
        -not $job.timed_out -and
        $job.status.ok -and
        $job.status.body.job.status -eq "succeeded" -and
        $job.result.ok -and
        $job.result.body -and
        $job.result.body.result -and
        $job.result.body.result.status -eq "success"
      )
    } catch {}

    $auditUpload.status = "ok"
    $auditUpload.ok = [bool]$auditUploadOk
    if (-not $auditUpload.ok) {
      throw "Audit job completed but did not return success result."
    }
  } catch {
    $auditUpload.status = "error"
    $auditUpload.error = $_.Exception.Message
  }
}
Write-Json (Join-Path $RunDir "17_audit_uploaded_data.json") $auditUpload

# --- LLM usage snapshot (after) ---
$llmUsageAfter = Safe-Request "GET" "$Backend/api/diagnostics/llm-usage"
Write-Json (Join-Path $RunDir "18_llm_usage_after.json") $llmUsageAfter

# --- Summary ---
$summary = [ordered]@{
  run_id = $Stamp
  backend = $Backend
  integrity_status = $integrity.status
  health_ok = $health.ok
  diagnostics_ok = $diag.ok
  auth_wrong_rejected_ok = $authChecks.wrong_rejected_ok
  auth_correct_ok = $authChecks.correct_ok_ok
  auth_correct_skipped = $authChecks.correct_skipped
  env_checked = (-not $envChecks.skipped)
  creative_ok = $creative.ok
  audit_expected_fail = ($audit.status -eq 400)
  audit_uploaded_data_ok = $auditUpload.ok
  advisor_checked = $advisor.ok
  sa360_checked = (-not $sa360.skipped)
  sa360_connected = ($sa360.oauth_status -and $sa360.oauth_status.body -and $sa360.oauth_status.body.connected -eq $true)
  sa360_fetch_ok = $null
  sa360_fetch_and_audit_ok = $null
  sa360_conversion_actions_ok = $null
  sa360_conversion_actions_count = $null
  route_latency_ms = $routerPerf.route_latency_ms
  route_latency_ok = $routerPerf.route_latency_ok
  route_verify_failed = $routerPerf.route_verify_failed
  default_account_saved = $routerPerf.default_account_saved
  plan_empty_ids_ok = $routerPerf.plan_empty_ids_ok
  plan_latency_ms = $routerPerf.latency_ms
  plan_latency_ok = ($routerPerf.latency_ms -ne $null -and $routerPerf.latency_ms -le $MaxPlanLatencyMs)
  router_verify_failed = $routerPerf.router_verify_failed
  chat_latency_ms = $chatLatency.latency_ms
  chat_latency_ok = $chatLatency.ok
  custom_metric_present = $routerPerf.custom_metric_present
  custom_metric_key = $routerPerf.custom_metric_key
  custom_metric_key_ok = $routerPerf.custom_metric_key_ok
  custom_metric_explicit_token_blocked = $null
  custom_metric_explicit_token_suggestions_count = $null
  custom_metric_explicit_token_latency_ms = $null
  custom_metric_relational_inferred = $null
  custom_metric_relational_key = $null
  custom_metric_relational_drivers_ok = $null
  custom_metric_relational_latency_ms = $null
  custom_metric_excluded_action_name = $null
  custom_metric_excluded_action_inferred = $null
  custom_metric_excluded_action_key = $null
  custom_metric_excluded_action_drivers_ok = $null
  custom_metric_excluded_action_uses_all_conversions = $null
  custom_metric_excluded_action_latency_ms = $null
  generic_relational_no_custom_ok = $null
  generic_relational_no_custom_latency_ms = $null
  audit_plan_and_run_ok = $null
  audit_plan_and_run_signature_error = $null
  pmax_ok = $pmax.ok
  serp_ok = $serp.ok
  competitor_ok = $comp.ok
  intel_ok = $intel.ok
  multisheet_upload_ok = $null
  spec_eval_status = $specEval.status
  spec_eval_summary = $specEval.summary
  sa360_parity_total = $null
  sa360_parity_passed = $null
  sa360_parity_ok = $null
  sa360_accuracy_total = $null
  sa360_accuracy_passed = $null
  sa360_accuracy_ok = $null
  llm_config = $llmUsageAfter.body.config
  azure_budget = $llmUsageAfter.body.azure_budget
  llm_usage_delta = $null
}
if ($routerPerf.custom_metric_explicit_token) {
  $summary.custom_metric_explicit_token_blocked = $routerPerf.custom_metric_explicit_token.blocked
  $summary.custom_metric_explicit_token_suggestions_count = $routerPerf.custom_metric_explicit_token.suggestions_count
  $summary.custom_metric_explicit_token_latency_ms = $routerPerf.custom_metric_explicit_token.latency_ms
}
if ($routerPerf.custom_metric_relational) {
  $summary.custom_metric_relational_inferred = $routerPerf.custom_metric_relational.inferred
  $summary.custom_metric_relational_key = $routerPerf.custom_metric_relational.metric_key
  $summary.custom_metric_relational_drivers_ok = $routerPerf.custom_metric_relational.drivers_ok
  $summary.custom_metric_relational_latency_ms = $routerPerf.custom_metric_relational.latency_ms
}
if ($routerPerf.custom_metric_excluded_action -and -not $routerPerf.custom_metric_excluded_action.skipped) {
  $summary.custom_metric_excluded_action_name = $routerPerf.custom_metric_excluded_action.name
  $summary.custom_metric_excluded_action_inferred = $routerPerf.custom_metric_excluded_action.inferred
  $summary.custom_metric_excluded_action_key = $routerPerf.custom_metric_excluded_action.metric_key
  $summary.custom_metric_excluded_action_drivers_ok = $routerPerf.custom_metric_excluded_action.drivers_ok
  $summary.custom_metric_excluded_action_uses_all_conversions = $routerPerf.custom_metric_excluded_action.uses_all_conversions
  $summary.custom_metric_excluded_action_latency_ms = $routerPerf.custom_metric_excluded_action.latency_ms
}
if ($routerPerf.generic_relational_no_custom) {
  $summary.generic_relational_no_custom_ok = $routerPerf.generic_relational_no_custom.ok
  $summary.generic_relational_no_custom_latency_ms = $routerPerf.generic_relational_no_custom.latency_ms
}
if ($routerPerf.audit_plan_and_run) {
  $summary.audit_plan_and_run_ok = $routerPerf.audit_plan_and_run.ok
  $summary.audit_plan_and_run_signature_error = $routerPerf.audit_plan_and_run.signature_error
}
if ($sa360.fetch) { $summary.sa360_fetch_ok = $sa360.fetch.ok }
if ($sa360.fetch_and_audit) { $summary.sa360_fetch_and_audit_ok = $sa360.fetch_and_audit.ok }
if ($sa360.conversion_actions) { 
  $summary.sa360_conversion_actions_ok = $sa360.conversion_actions.ok
  try {
    if ($sa360.conversion_actions.body -and $sa360.conversion_actions.body.actions_count) {
      $summary.sa360_conversion_actions_count = $sa360.conversion_actions.body.actions_count
    }
  } catch {}
}
if ($multisheet.status -eq "ok" -and $multisheet.summary) {
  try { $summary.multisheet_upload_ok = [bool]$multisheet.summary.ok } catch {}
}
if ($specEval.status -eq "ok" -and $specEval.summary -and $specEval.summary.sa360_parity) {
  try { $summary.sa360_parity_total = [int]$specEval.summary.sa360_parity.total } catch {}
  try { $summary.sa360_parity_passed = [int]$specEval.summary.sa360_parity.passed } catch {}
  if ($summary.sa360_parity_total -ne $null -and $summary.sa360_parity_total -gt 0) {
    $summary.sa360_parity_ok = ($summary.sa360_parity_passed -eq $summary.sa360_parity_total)
  }
}
if ($specEval.status -eq "ok" -and $specEval.summary -and $specEval.summary.sa360_accuracy) {
  try { $summary.sa360_accuracy_total = [int]$specEval.summary.sa360_accuracy.total } catch {}
  try { $summary.sa360_accuracy_passed = [int]$specEval.summary.sa360_accuracy.passed } catch {}
  if ($summary.sa360_accuracy_total -ne $null -and $summary.sa360_accuracy_total -gt 0) {
    $summary.sa360_accuracy_ok = ($summary.sa360_accuracy_passed -eq $summary.sa360_accuracy_total)
  }
}

try {
  $before = $llmUsageBefore.body.usage
  $after = $llmUsageAfter.body.usage
  if ($before -and $after) {
    $resetDetected = $false
    foreach ($k in @("local_success", "local_error", "azure_success", "azure_error")) {
      try {
        if ([int]$after.$k -lt [int]$before.$k) { $resetDetected = $true }
      } catch {}
    }
    if ($resetDetected) {
      $summary.llm_usage_delta = [ordered]@{
        reset_detected = $true
        note = "LLM usage counters decreased during run (likely service restart); deltas are not comparable."
      }
    } else {
      $summary.llm_usage_delta = [ordered]@{
        reset_detected = $false
        local_success = ([int]$after.local_success - [int]$before.local_success)
        local_error = ([int]$after.local_error - [int]$before.local_error)
        azure_success = ([int]$after.azure_success - [int]$before.azure_success)
        azure_error = ([int]$after.azure_error - [int]$before.azure_error)
      }
    }
  }
} catch {}
Write-Json (Join-Path $RunDir "summary.json") $summary

Write-Host "Full QA run folder: $RunDir"
Write-Host "Summary:"
$summary | ConvertTo-Json -Depth 4
