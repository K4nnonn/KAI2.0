param(
  [string]$Backend = $env:KAI_BACKEND_URL,
  [string]$SessionId = $env:KAI_QA_SESSION_ID,
  [string]$TargetCustomerId = $env:KAI_QA_TARGET_CUSTOMER_ID,
  [string]$SpecDir = "",
  [string]$OutDir = "",
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Backend)) {
  $Backend = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}
if ([string]::IsNullOrWhiteSpace($SessionId)) {
  throw "SessionId is required (set KAI_QA_SESSION_ID or pass -SessionId)."
}
if ([string]::IsNullOrWhiteSpace($TargetCustomerId)) {
  throw "TargetCustomerId is required (set KAI_QA_TARGET_CUSTOMER_ID or pass -TargetCustomerId)."
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
if ([string]::IsNullOrWhiteSpace($SpecDir)) {
  $SpecDir = Join-Path $PSScriptRoot "specs"
}
if (-not (Test-Path $SpecDir)) {
  throw "SpecDir not found: $SpecDir"
}

function Redact-JsonText($Json) {
  if ([string]::IsNullOrWhiteSpace($Json)) { return $Json }
  $redacted = $Json
  $queryKeys = @(
    "session_id","admin_password","code","access_token","refresh_token","client_secret",
    "sig","se","sp","sv","sr","st","skoid","sktid","skt","ske","skv","sip","si"
  )
  foreach ($k in $queryKeys) {
    $pattern = '((?:\\?|&|\\\\u0026)' + [regex]::Escape($k) + '=).*?(?=(?:&|\\\\u0026|"))'
    $redacted = [regex]::Replace($redacted, $pattern, '$1REDACTED')
  }
  # Redact direct secret-like fields if they appear as JSON keys (share-safe artifacts).
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

function Write-Json($Path, $Obj) {
  $json = $Obj | ConvertTo-Json -Depth 14
  $json = Redact-JsonText $json
  $json | Set-Content -Path $Path -Encoding UTF8
}

function Substitute($Value, $Vars) {
  if ($Value -eq $null) { return $null }
  if ($Value -is [string]) {
    $s = $Value
    foreach ($k in $Vars.Keys) {
      $s = $s.Replace('${' + $k + '}', [string]$Vars[$k])
    }
    return $s
  }
  if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [hashtable]) -and -not ($Value -is [pscustomobject])) {
    $out = @()
    foreach ($x in $Value) {
      $out += Substitute $x $Vars
    }
    # Prevent PowerShell from unrolling single-item arrays into scalars on return.
    return ,$out
  }
  # Convert objects/dictionaries into PSCustomObject so downstream loops that rely on
  # .PSObject.Properties behave consistently (avoid OrderedDictionary.Keys/Values noise).
  if ($Value -is [System.Collections.IDictionary] -or $Value -is [pscustomobject]) {
    $out = [ordered]@{}
    if ($Value -is [System.Collections.IDictionary]) {
      foreach ($k in $Value.Keys) {
        $out[[string]$k] = Substitute $Value[$k] $Vars
      }
    } else {
      foreach ($p in $Value.PSObject.Properties) {
        $out[$p.Name] = Substitute $p.Value $Vars
      }
    }
    return [pscustomobject]$out
  }
  return $Value
}

function Get-PathValue($Obj, [string]$Path) {
  if ($Obj -eq $null) { return $null }
  $cur = $Obj
  foreach ($part in ($Path -split '\.')) {
    if ($cur -eq $null) { return $null }
    # Support numeric indexing for JSON arrays (e.g. "actions.0.metric_key").
    if (($cur -is [System.Array] -or $cur -is [System.Collections.IList]) -and ($part -match '^\d+$')) {
      $idx = [int]$part
      if ($idx -lt 0 -or $idx -ge $cur.Count) { return $null }
      $cur = $cur[$idx]
      continue
    }
    # Support PowerShell dictionaries, including [ordered]@{} (OrderedDictionary) and hashtables.
    if ($cur -is [System.Collections.IDictionary]) {
      if ($cur.Contains($part)) { $cur = $cur[$part] } else { return $null }
    } else {
      $prop = $cur.PSObject.Properties[$part]
      if ($prop) { $cur = $prop.Value } else { return $null }
    }
  }
  # Preserve arrays as arrays: PowerShell enumerates arrays by default when returning from functions,
  # which breaks assertions that need to reason about array length.
  Write-Output -NoEnumerate $cur
}

function Get-AnyPathValue($Resp, [string]$Path) {
  if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
  # Preserve arrays as arrays: PowerShell otherwise unrolls single-item arrays into scalars,
  # which breaks assertions like array_min_len. Use -NoEnumerate to force a single return object.
  $val = $null
  if ($Path.StartsWith("__")) {
    $val = Get-PathValue $Resp ($Path.Substring(2))
  } else {
    $val = Get-PathValue $Resp.body $Path
  }
  Write-Output -NoEnumerate $val
}

function Safe-Post($Url, $BodyObj) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $status = $null
  $ok = $false
  $body = $null
  $err = $null
  try {
    $resp = Invoke-WebRequest -Method POST -Uri $Url -ContentType "application/json" -Body ($BodyObj | ConvertTo-Json -Depth 10) -UseBasicParsing
    $status = $resp.StatusCode
    $ok = ($status -ge 200 -and $status -lt 300)
    if ($resp.Content) { $body = $resp.Content | ConvertFrom-Json }
  } catch {
    $err = $_.Exception.Message
    if ($_.Exception.Response) {
      $status = [int]$_.Exception.Response.StatusCode
      try {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $content = $reader.ReadToEnd()
        try { $body = $content | ConvertFrom-Json } catch { $body = $content }
      } catch {}
    }
  } finally {
    $sw.Stop()
  }
  return [ordered]@{
    url = $Url
    method = "POST"
    status = $status
    ok = $ok
    latency_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 2)
    error = $err
    body = $body
  }
}

function Safe-Get($Url) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $status = $null
  $ok = $false
  $body = $null
  $err = $null
  try {
    $resp = Invoke-WebRequest -Method GET -Uri $Url -UseBasicParsing
    $status = $resp.StatusCode
    $ok = ($status -ge 200 -and $status -lt 300)
    if ($resp.Content) { $body = $resp.Content | ConvertFrom-Json }
  } catch {
    $err = $_.Exception.Message
    if ($_.Exception.Response) {
      $status = [int]$_.Exception.Response.StatusCode
      try {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $content = $reader.ReadToEnd()
        try { $body = $content | ConvertFrom-Json } catch { $body = $content }
      } catch {}
    }
  } finally {
    $sw.Stop()
  }
  return [ordered]@{
    url = $Url
    method = "GET"
    status = $status
    ok = $ok
    latency_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 2)
    error = $err
    body = $body
  }
}

function NearlyEqual([double]$A, [double]$B, [double]$Tol) {
  return ([math]::Abs($A - $B) -le $Tol)
}

function Validate-Spec($Spec, [string]$FileName) {
  $errs = @()
  if (-not $Spec) { return @("spec_parse_failed") }
  if ([string]::IsNullOrWhiteSpace([string]$Spec.id)) { $errs += "missing_id" }
  if ([string]::IsNullOrWhiteSpace([string]$Spec.pillar)) { $errs += "missing_pillar" }
  if ([string]::IsNullOrWhiteSpace([string]$Spec.kind)) { $errs += "missing_kind" }
  if (-not $Spec.request) { $errs += "missing_request" }
  if (-not $Spec.expect) { $errs += "missing_expect" }

  $pillar = ([string]$Spec.pillar).ToLower().Trim()
  if ($pillar -and ($pillar -notin @("intent","purpose","quality","functionality"))) {
    $errs += "invalid_pillar:$pillar"
  }

  $kind = ([string]$Spec.kind).ToLower().Trim()
  if ($kind -and ($kind -notin @("route","plan_and_run","advisor_diag","send","get"))) {
    $errs += "invalid_kind:$kind"
  }

  if ($Spec.intent_goal) {
    $target = ""
    try { $target = ([string]$Spec.intent_goal.target).ToLower().Trim() } catch {}
    if ($target -and ($target -notin @("baseline", "followup_chat_send"))) {
      $errs += "invalid_intent_goal_target:$target"
    }
    if ($target -eq "followup_chat_send") {
      $hasFollowup = $false
      try {
        $hasFollowup = [bool]($Spec.followup -and $Spec.followup.chat_send)
      } catch { $hasFollowup = $false }
      if (-not $hasFollowup) {
        $errs += "intent_goal_followup_target_without_followup_chat_send"
      }
    }
  }

  return $errs
}

function Pctl($Arr, [double]$P) {
  if (-not $Arr -or $Arr.Count -eq 0) { return $null }
  $sorted = $Arr | Sort-Object
  $idx = [math]::Floor($P * ($sorted.Count - 1))
  if ($idx -lt 0) { $idx = 0 }
  if ($idx -ge $sorted.Count) { $idx = $sorted.Count - 1 }
  return [double]$sorted[$idx]
}

function Parse-MetricFromSummary([string]$Summary, [string]$Label) {
  if ([string]::IsNullOrWhiteSpace($Summary)) { return $null }
  switch ($Label) {
    "conversions" {
      # PowerShell single-quoted strings do not treat backslash as an escape. Use \s for whitespace (not \\s).
      if ($Summary -match 'Conversions:\s*([0-9,]+(?:\.[0-9]+)?)') { return [double]($matches[1] -replace ',', '') }
    }
    "cost" {
      if ($Summary -match 'Cost:\s*\$([0-9,]+(?:\.[0-9]+)?)') { return [double]($matches[1] -replace ',', '') }
    }
    "cpc" {
      if ($Summary -match 'CPC:\s*\$([0-9,]+(?:\.[0-9]+)?)') { return [double]($matches[1] -replace ',', '') }
    }
    "ctr" {
      if ($Summary -match 'CTR:\s*([0-9,]+(?:\.[0-9]+)?)(?:%)?') { return [double]($matches[1] -replace ',', '') }
    }
  }
  return $null
}

function Get-StringArray($Value) {
  if ($Value -eq $null) { return @() }
  if ($Value -is [string]) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return @() }
    return @($Value)
  }
  if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [hashtable]) -and -not ($Value -is [pscustomobject])) {
    $arr = @()
    foreach ($x in $Value) {
      if ($x -eq $null) { continue }
      $s = [string]$x
      if (-not [string]::IsNullOrWhiteSpace($s)) { $arr += $s }
    }
    return $arr
  }
  $single = [string]$Value
  if ([string]::IsNullOrWhiteSpace($single)) { return @() }
  return @($single)
}

function Get-CombinedTextFromPaths($Resp, $Paths) {
  $parts = @()
  foreach ($p in @($Paths)) {
    $path = [string]$p
    if ([string]::IsNullOrWhiteSpace($path)) { continue }
    $val = Get-AnyPathValue $Resp $path
    if ($val -eq $null) { continue }
    if ($val -is [string]) {
      if (-not [string]::IsNullOrWhiteSpace($val)) { $parts += $val }
      continue
    }
    if ($val -is [System.Array] -or $val -is [System.Collections.IList]) {
      foreach ($item in @($val)) {
        if ($item -eq $null) { continue }
        $s = [string]$item
        if (-not [string]::IsNullOrWhiteSpace($s)) { $parts += $s }
      }
      continue
    }
    try {
      $s = ($val | ConvertTo-Json -Depth 6 -Compress)
      if (-not [string]::IsNullOrWhiteSpace($s)) { $parts += $s }
    } catch {
      $s = [string]$val
      if (-not [string]::IsNullOrWhiteSpace($s)) { $parts += $s }
    }
  }
  return ($parts -join "`n")
}

function Evaluate-IntentGoalAlignment($Resp, $Cfg) {
  $out = [ordered]@{
    ok = $true
    checks = @()
    response_paths = @()
    response_text_len = 0
  }
  if (-not $Cfg) { return $out }

  $intentPath = [string]$Cfg.intent_path
  if ([string]::IsNullOrWhiteSpace($intentPath)) { $intentPath = "intent" }
  $allowedIntents = Get-StringArray $Cfg.allowed_intents
  if ($allowedIntents.Count -gt 0) {
    $actualIntent = [string](Get-AnyPathValue $Resp $intentPath)
    $passIntent = ($allowedIntents | ForEach-Object { $_.ToLower().Trim() }) -contains ($actualIntent.ToLower().Trim())
    $out.checks += [ordered]@{
      type = "allowed_intents"
      path = $intentPath
      expected = $allowedIntents
      actual = $actualIntent
      ok = $passIntent
    }
    if (-not $passIntent) { $out.ok = $false }
  }

  $responsePaths = Get-StringArray $Cfg.response_paths
  if ($responsePaths.Count -eq 0) {
    $responsePaths = @("__body.reply", "summary", "analysis.summary")
  }
  $out.response_paths = $responsePaths
  $text = Get-CombinedTextFromPaths $Resp $responsePaths
  $out.response_text_len = ($text | Out-String).Length
  $lower = $text.ToLower()

  $mustAll = Get-StringArray $Cfg.must_include_all
  foreach ($tok in $mustAll) {
    $needle = $tok.ToLower()
    $pass = $lower.Contains($needle)
    $out.checks += [ordered]@{
      type = "must_include_all"
      token = $tok
      ok = $pass
    }
    if (-not $pass) { $out.ok = $false }
  }

  $mustAny = Get-StringArray $Cfg.must_include_any
  if ($mustAny.Count -gt 0) {
    $hits = @()
    foreach ($tok in $mustAny) {
      $needle = $tok.ToLower()
      if ($lower.Contains($needle)) { $hits += $tok }
    }
    $minAny = 1
    try {
      if ($Cfg.min_any_matches -ne $null) { $minAny = [int]$Cfg.min_any_matches }
    } catch {}
    if ($minAny -lt 1) { $minAny = 1 }
    $passAny = ($hits.Count -ge $minAny)
    $out.checks += [ordered]@{
      type = "must_include_any"
      min = $minAny
      expected = $mustAny
      hits = $hits
      ok = $passAny
    }
    if (-not $passAny) { $out.ok = $false }
  }

  $mustNot = Get-StringArray $Cfg.must_not_include
  if ($mustNot.Count -gt 0) {
    $bad = @()
    foreach ($tok in $mustNot) {
      $needle = $tok.ToLower()
      if ($lower.Contains($needle)) { $bad += $tok }
    }
    $passNot = ($bad.Count -eq 0)
    $out.checks += [ordered]@{
      type = "must_not_include"
      forbidden = $mustNot
      hits = $bad
      ok = $passNot
    }
    if (-not $passNot) { $out.ok = $false }
  }

  return $out
}

function Get-CapabilityTags($Spec, [string]$Pillar, [string]$Kind) {
  $caps = @()
  try {
    $caps = Get-StringArray $Spec.capabilities
  } catch {
    $caps = @()
  }
  if ($caps.Count -eq 0) {
    $caps = @("pillar:$Pillar", "kind:$Kind")
  }
  return @(
    $caps |
      ForEach-Object { ([string]$_).Trim().ToLower() } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
      Select-Object -Unique
  )
}

function Get-EvidenceStats($Checks, $Assertions, [bool]$SpecOk) {
  $total = 0
  $passed = 0
  foreach ($c in @($Checks)) {
    $total += 1
    try { if ([bool]$c.ok) { $passed += 1 } } catch {}
  }

  foreach ($k in @($Assertions.Keys)) {
    $v = $Assertions[$k]
    if ($v -isnot [System.Collections.IDictionary]) { continue }

    # Metamorphic checks contribute one evidence point per variant.
    if ($k -eq "metamorphic" -and $v.Contains("total") -and $v.Contains("failed")) {
      $mTotal = [int]$v["total"]
      $mFailed = [int]$v["failed"]
      if ($mTotal -gt 0) {
        $total += $mTotal
        $passed += ($mTotal - $mFailed)
      }
      continue
    }

    if ($v.Contains("checks")) {
      $arr = @($v["checks"])
      if ($arr.Count -gt 0) {
        foreach ($i in $arr) {
          $total += 1
          $itemOk = $false
          try { $itemOk = [bool]$i.ok } catch { $itemOk = $false }
          if ($itemOk) { $passed += 1 }
        }
        continue
      }
    }

    if ($v.Contains("ok")) {
      $total += 1
      try { if ([bool]$v["ok"]) { $passed += 1 } } catch {}
    }
  }

  if ($total -le 0) {
    $total = 1
    if ($SpecOk) { $passed = 1 } else { $passed = 0 }
  }
  $rate = [math]::Round(($passed / $total), 4)
  return [ordered]@{
    total = $total
    passed = $passed
    failed = ($total - $passed)
    pass_rate = $rate
  }
}

function Get-WilsonLowerBound([int]$Successes, [int]$Trials, [double]$Z = 1.96) {
  if ($Trials -le 0) { return 0.0 }
  $p = [double]$Successes / [double]$Trials
  $z2 = $Z * $Z
  $den = 1.0 + ($z2 / $Trials)
  $center = $p + ($z2 / (2.0 * $Trials))
  $margin = $Z * [math]::Sqrt((($p * (1.0 - $p)) + ($z2 / (4.0 * $Trials))) / $Trials)
  $lb = ($center - $margin) / $den
  if ($lb -lt 0.0) { $lb = 0.0 }
  if ($lb -gt 1.0) { $lb = 1.0 }
  return [math]::Round($lb, 4)
}

function Get-ConfidenceLevel([double]$Wilson95LowerBound, [int]$EvidenceTrials) {
  if ($EvidenceTrials -ge 20 -and $Wilson95LowerBound -ge 0.90) { return "high" }
  if ($EvidenceTrials -ge 10 -and $Wilson95LowerBound -ge 0.75) { return "medium" }
  return "low"
}

function Assert-SummaryMatchesResult($Body, $Metrics) {
  $out = [ordered]@{
    ok = $true
    checks = @()
  }
  $summary = [string]$Body.summary
  foreach ($m in $Metrics) {
    $parsed = Parse-MetricFromSummary $summary $m
    $actual = $null
    try { $actual = [double](Get-PathValue $Body ("result.current." + $m)) } catch {}
    $ok = $false
    if ($parsed -ne $null -and $actual -ne $null) {
      $tol = 0.02
      if ($m -eq "cost") { $tol = 0.50 }
      $ok = NearlyEqual $parsed ([math]::Round($actual, 2)) $tol
    }
    $out.checks += [ordered]@{
      metric = $m
      parsed = $parsed
      expected = $actual
      ok = $ok
    }
    if (-not $ok) { $out.ok = $false }
  }
  return $out
}

function Get-DefaultMetricTolerance([string]$Metric) {
  $m = ([string]$Metric).ToLower().Trim()
  if ($m -in @("impressions", "clicks", "conversions")) { return 1.0 }
  if ($m -in @("cost", "conversions_value")) { return 1.0 }
  if ($m -in @("ctr", "cpc", "cvr", "cpa", "roas")) { return 0.05 }
  if ($m.StartsWith("custom:")) { return 0.5 }
  return 0.05
}

function Get-ConfigTolerance($Cfg, [string]$Metric) {
  $tol = $null
  try {
    if ($Cfg -and $Cfg.abs_tolerance) {
      foreach ($p in $Cfg.abs_tolerance.PSObject.Properties) {
        if ([string]$p.Name -eq [string]$Metric) {
          $tol = [double]$p.Value
          break
        }
      }
    }
  } catch {}
  if ($tol -eq $null) { $tol = Get-DefaultMetricTolerance $Metric }
  return [double]$tol
}

function Get-Sa360AssertContext($Resp, $RequestBody, $Cfg, [string]$DefaultSessionId) {
  $ctx = [ordered]@{
    ok = $true
    reason = $null
    session_id = ""
    customer_id = ""
    result = $null
    current_range = ""
    previous_range = ""
    compare_previous = $true
    source = "current_live"
    metrics = @("impressions","clicks","cost","conversions","ctr","cpc")
    report_names = @()
  }

  $executed = $false
  try { $executed = [bool]$Resp.body.executed } catch {}
  if (-not $executed) {
    $ctx.ok = $false
    $ctx.reason = "plan_not_executed"
    return $ctx
  }

  try { $ctx.result = $Resp.body.result } catch {}
  if (-not $ctx.result) {
    $ctx.ok = $false
    $ctx.reason = "missing_result"
    return $ctx
  }

  $mode = ""
  try { $mode = [string]$ctx.result.mode } catch {}
  if ($mode -ne "performance") {
    $ctx.ok = $false
    $ctx.reason = "non_performance_mode"
    return $ctx
  }

  try {
    if ($RequestBody -and $RequestBody.session_id) { $ctx.session_id = [string]$RequestBody.session_id }
  } catch {}
  if ([string]::IsNullOrWhiteSpace($ctx.session_id)) { $ctx.session_id = $DefaultSessionId }

  try {
    if ($RequestBody -and $RequestBody.customer_ids -and @($RequestBody.customer_ids).Count -gt 0) {
      $ctx.customer_id = [string]@($RequestBody.customer_ids)[0]
    }
  } catch {}
  if ([string]::IsNullOrWhiteSpace($ctx.customer_id)) {
    try {
      if ($Resp.body.plan -and $Resp.body.plan.customer_ids -and @($Resp.body.plan.customer_ids).Count -gt 0) {
        $ctx.customer_id = [string]@($Resp.body.plan.customer_ids)[0]
      }
    } catch {}
  }
  if ([string]::IsNullOrWhiteSpace($ctx.customer_id)) {
    $ctx.ok = $false
    $ctx.reason = "missing_customer_id"
    return $ctx
  }

  try { $ctx.current_range = [string]$ctx.result.date_range_current } catch {}
  if ([string]::IsNullOrWhiteSpace($ctx.current_range)) {
    try { $ctx.current_range = [string]$RequestBody.date_range } catch {}
  }
  if ([string]::IsNullOrWhiteSpace($ctx.current_range)) {
    $ctx.ok = $false
    $ctx.reason = "missing_current_date_range"
    return $ctx
  }

  try { $ctx.previous_range = [string]$ctx.result.date_range_previous } catch {}

  try {
    if ($Cfg -and $Cfg.compare_previous -ne $null) {
      $ctx.compare_previous = [bool]$Cfg.compare_previous
    } elseif ([string]::IsNullOrWhiteSpace($ctx.previous_range)) {
      $ctx.compare_previous = $false
    }
  } catch {
    if ([string]::IsNullOrWhiteSpace($ctx.previous_range)) { $ctx.compare_previous = $false }
  }

  try {
    if ($Cfg -and $Cfg.source) { $ctx.source = [string]$Cfg.source }
  } catch {}
  if ([string]::IsNullOrWhiteSpace($ctx.source)) { $ctx.source = "current_live" }

  try {
    if ($Cfg -and $Cfg.metrics) {
      $cfgMetrics = @($Cfg.metrics) | ForEach-Object { [string]$_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
      if ($cfgMetrics.Count -gt 0) { $ctx.metrics = $cfgMetrics }
    }
  } catch {}

  $reportNames = @()
  try {
    if ($Cfg -and $Cfg.report_names) {
      $reportNames = Get-StringArray $Cfg.report_names
    }
  } catch {}
  if ($reportNames.Count -eq 0) {
    $debugFrame = ""
    try { $debugFrame = [string](Get-PathValue $ctx.result "debug.frame") } catch {}
    if (-not [string]::IsNullOrWhiteSpace($debugFrame)) { $reportNames += $debugFrame }
    $hasCustomMetric = $false
    foreach ($metric in $ctx.metrics) {
      if (([string]$metric).ToLower().StartsWith("custom:")) {
        $hasCustomMetric = $true
        break
      }
    }
    if ($hasCustomMetric) { $reportNames += "conversion_action_summary" }
  }
  if ($reportNames.Count -eq 0) { $reportNames = @("customer_performance") }
  $ctx.report_names = @(
    $reportNames |
      ForEach-Object { ([string]$_).Trim().ToLower() } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
      Select-Object -Unique
  )

  return $ctx
}

function Assert-Sa360Parity($Resp, $RequestBody, $Cfg, [string]$BackendUrl, [string]$DefaultSessionId) {
  $out = [ordered]@{
    ok = $true
    checks = @()
    source = "current_live"
    compare_previous = $true
    diagnostics = $null
    request = $null
    reason = $null
  }

  $ctx = Get-Sa360AssertContext $Resp $RequestBody $Cfg $DefaultSessionId
  if (-not $ctx.ok) {
    $out.ok = $false
    $out.reason = $ctx.reason
    return $out
  }

  $result = $ctx.result
  $sid = $ctx.session_id
  $cid = $ctx.customer_id
  $currentRange = $ctx.current_range
  $previousRange = $ctx.previous_range
  $comparePrevious = [bool]$ctx.compare_previous
  $source = [string]$ctx.source
  $metrics = @($ctx.metrics)
  $reportNames = @($ctx.report_names)

  $out.compare_previous = $comparePrevious
  $out.source = $source

  $diagReq = [ordered]@{
    session_id = $sid
    customer_ids = @($cid)
    date_range = $currentRange
    include_previous = $comparePrevious
    report_names = $reportNames
    metrics = $metrics
    bypass_cache = $true
    async_mode = $false
  }
  $out.request = $diagReq

  $diagUrl = "$BackendUrl/api/diagnostics/sa360/perf-window"
  $diagResp = Safe-Post $diagUrl $diagReq
  $out.diagnostics = [ordered]@{
    ok = $diagResp.ok
    status = $diagResp.status
    error = $diagResp.error
    date_range_current = $null
    date_range_previous = $null
  }
  try { $out.diagnostics.date_range_current = $diagResp.body.date_range_current } catch {}
  try { $out.diagnostics.date_range_previous = $diagResp.body.date_range_previous } catch {}

  if (-not $diagResp.ok) {
    $out.ok = $false
    $out.reason = "diagnostics_request_failed"
    return $out
  }

  $snapCurrent = $null
  try { $snapCurrent = Get-PathValue $diagResp.body ("snapshots." + $source + ".metrics") } catch {}
  if (-not $snapCurrent) {
    $out.ok = $false
    $out.reason = "missing_source_snapshot_metrics"
    return $out
  }

  foreach ($m in $metrics) {
    $planVal = $null
    $liveVal = $null
    $okMetric = $false
    $delta = $null
    $tol = Get-ConfigTolerance $Cfg $m

    $planRaw = $null
    $liveRaw = $null
    try { $planRaw = Get-PathValue $result ("current." + $m) } catch {}
    try { $liveRaw = Get-PathValue $snapCurrent $m } catch {}
    # Important: do NOT cast $null to [double] (PowerShell casts $null -> 0.0), which would create
    # false-positive parity passes for missing metrics.
    if ($planRaw -ne $null) { try { $planVal = [double]$planRaw } catch { $planVal = $null } }
    if ($liveRaw -ne $null) { try { $liveVal = [double]$liveRaw } catch { $liveVal = $null } }

    if ($planVal -ne $null -and $liveVal -ne $null) {
      $delta = ($planVal - $liveVal)
      $okMetric = NearlyEqual $planVal $liveVal $tol
    }

    $row = [ordered]@{
      window = "current"
      metric = $m
      plan = $planVal
      source = $liveVal
      delta = $delta
      tolerance_abs = $tol
      ok = $okMetric
    }
    $out.checks += $row
    if (-not $okMetric) { $out.ok = $false }
  }

  if ($comparePrevious) {
    $snapPrev = $null
    try { $snapPrev = Get-PathValue $diagResp.body ("snapshots.previous_live.metrics") } catch {}
    $planPrev = $null
    try { $planPrev = $result.previous } catch {}
    if ($snapPrev -and $planPrev) {
      foreach ($m in $metrics) {
        $planVal = $null
        $liveVal = $null
        $okMetric = $false
        $delta = $null
        $tol = Get-ConfigTolerance $Cfg $m

        $planRaw = $null
        $liveRaw = $null
        try { $planRaw = Get-PathValue $result ("previous." + $m) } catch {}
        try { $liveRaw = Get-PathValue $snapPrev $m } catch {}
        if ($planRaw -ne $null) { try { $planVal = [double]$planRaw } catch { $planVal = $null } }
        if ($liveRaw -ne $null) { try { $liveVal = [double]$liveRaw } catch { $liveVal = $null } }

        if ($planVal -ne $null -and $liveVal -ne $null) {
          $delta = ($planVal - $liveVal)
          $okMetric = NearlyEqual $planVal $liveVal $tol
        }

        $row = [ordered]@{
          window = "previous"
          metric = $m
          plan = $planVal
          source = $liveVal
          delta = $delta
          tolerance_abs = $tol
          ok = $okMetric
        }
        $out.checks += $row
        if (-not $okMetric) { $out.ok = $false }
      }
    } else {
      $out.ok = $false
      $out.reason = "previous_window_requested_but_missing"
    }
  }

  return $out
}

function Assert-Sa360Accuracy($Resp, $RequestBody, $Cfg, [string]$BackendUrl, [string]$DefaultSessionId) {
  $out = [ordered]@{
    ok = $true
    checks = @()
    source = "current_live"
    compare_previous = $true
    diagnostics = $null
    request = $null
    reason = $null
  }

  $ctx = Get-Sa360AssertContext $Resp $RequestBody $Cfg $DefaultSessionId
  if (-not $ctx.ok) {
    $out.ok = $false
    $out.reason = $ctx.reason
    return $out
  }

  $result = $ctx.result
  $sid = $ctx.session_id
  $cid = $ctx.customer_id
  $currentRange = $ctx.current_range
  $comparePrevious = [bool]$ctx.compare_previous
  $source = [string]$ctx.source
  $metrics = @($ctx.metrics)
  $reportNames = @($ctx.report_names)

  $out.compare_previous = $comparePrevious
  $out.source = $source

  $diagReq = [ordered]@{
    session_id = $sid
    customer_ids = @($cid)
    date_range = $currentRange
    include_previous = $comparePrevious
    report_names = $reportNames
    metrics = $metrics
    bypass_cache = $true
    async_mode = $false
  }
  $out.request = $diagReq

  $diagUrl = "$BackendUrl/api/diagnostics/sa360/perf-window"
  $diagResp = Safe-Post $diagUrl $diagReq
  $diagBody = $null
  if ($diagResp.ok) { $diagBody = $diagResp.body }

  $echoReports = @()
  try { $echoReports = @($diagBody.report_names | ForEach-Object { ([string]$_).Trim().ToLower() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) } catch {}
  $echoReports = @($echoReports | Select-Object -Unique)

  $snap = $null
  try { $snap = Get-PathValue $diagBody ("snapshots." + $source) } catch {}
  $snapMetrics = $null
  try { $snapMetrics = Get-PathValue $diagBody ("snapshots." + $source + ".metrics") } catch {}
  $snapDebugFrame = ""
  try { $snapDebugFrame = [string](Get-PathValue $diagBody ("snapshots." + $source + ".debug.frame")) } catch {}
  $snapDebugFrame = $snapDebugFrame.Trim().ToLower()

  $out.diagnostics = [ordered]@{
    ok = $diagResp.ok
    status = $diagResp.status
    error = $diagResp.error
    report_names_echo = $echoReports
    source_debug_frame = $snapDebugFrame
    date_range_current = $null
    date_range_previous = $null
  }
  try { $out.diagnostics.date_range_current = $diagBody.date_range_current } catch {}
  try { $out.diagnostics.date_range_previous = $diagBody.date_range_previous } catch {}

  if (-not $diagResp.ok) {
    $out.ok = $false
    $out.reason = "diagnostics_request_failed"
    return $out
  }

  $requireEcho = $true
  $requireFrameInReports = $true
  $requireMetricPresence = $true
  $maxLiveCachedCostDelta = $null
  try { if ($Cfg -and $Cfg.require_report_names_echo -ne $null) { $requireEcho = [bool]$Cfg.require_report_names_echo } } catch {}
  try { if ($Cfg -and $Cfg.require_source_frame_in_reports -ne $null) { $requireFrameInReports = [bool]$Cfg.require_source_frame_in_reports } } catch {}
  try { if ($Cfg -and $Cfg.require_metric_presence -ne $null) { $requireMetricPresence = [bool]$Cfg.require_metric_presence } } catch {}
  try { if ($Cfg -and $Cfg.max_live_cached_cost_delta_abs -ne $null) { $maxLiveCachedCostDelta = [double]$Cfg.max_live_cached_cost_delta_abs } } catch {}

  if ($requireEcho) {
    $requested = @($reportNames | ForEach-Object { ([string]$_).Trim().ToLower() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
    $missing = @()
    foreach ($r in $requested) {
      if ($echoReports -notcontains $r) { $missing += $r }
    }
    $row = [ordered]@{
      check = "report_names_echo"
      requested = $requested
      echoed = $echoReports
      missing = $missing
      ok = ($missing.Count -eq 0 -and $echoReports.Count -gt 0)
    }
    $out.checks += $row
    if (-not $row.ok) { $out.ok = $false }
  }

  if ($requireFrameInReports) {
    $inReports = $false
    if (-not [string]::IsNullOrWhiteSpace($snapDebugFrame)) {
      $inReports = ($echoReports -contains $snapDebugFrame)
      if (-not $inReports -and $reportNames.Count -gt 0) {
        $normalizedRequested = @($reportNames | ForEach-Object { ([string]$_).Trim().ToLower() } | Select-Object -Unique)
        $inReports = ($normalizedRequested -contains $snapDebugFrame)
      }
    }
    $row = [ordered]@{
      check = "source_frame_alignment"
      source = $source
      source_debug_frame = $snapDebugFrame
      echoed_report_names = $echoReports
      ok = $inReports
    }
    $out.checks += $row
    if (-not $row.ok) { $out.ok = $false }
  }

  if ($requireMetricPresence) {
    foreach ($m in $metrics) {
      $planVal = $null
      $sourceVal = $null
      $present = $false
      try { $planVal = Get-PathValue $result ("current." + $m) } catch {}
      try { $sourceVal = Get-PathValue $snapMetrics $m } catch {}
      if ($planVal -ne $null -and $sourceVal -ne $null) { $present = $true }
      $row = [ordered]@{
        check = "metric_presence"
        window = "current"
        metric = $m
        plan_present = ($planVal -ne $null)
        source_present = ($sourceVal -ne $null)
        ok = $present
      }
      $out.checks += $row
      if (-not $row.ok) { $out.ok = $false }
    }
  }

  if ($maxLiveCachedCostDelta -ne $null) {
    $delta = $null
    try { $delta = [double](Get-PathValue $diagBody "comparisons.current_cost_gap.delta") } catch {}
    $pass = $false
    if ($delta -ne $null) {
      $pass = ([math]::Abs($delta) -le [double]$maxLiveCachedCostDelta)
    }
    $row = [ordered]@{
      check = "cached_live_cost_gap"
      delta = $delta
      tolerance_abs = [double]$maxLiveCachedCostDelta
      ok = $pass
    }
    $out.checks += $row
    if (-not $row.ok) { $out.ok = $false }
  }

  if (-not $out.ok -and [string]::IsNullOrWhiteSpace($out.reason)) {
    $out.reason = "accuracy_checks_failed"
  }
  return $out
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $RunBase = Join-Path $RepoRoot "verification_runs"
  New-Item -ItemType Directory -Force -Path $RunBase | Out-Null
  $OutDir = Join-Path $RunBase ("spec_eval_" + $Stamp)
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$TraceDir = Join-Path $OutDir "traces"
New-Item -ItemType Directory -Force -Path $TraceDir | Out-Null

# Precondition: ensure the QA session has a default SA360 account saved.
# Several routing specs intentionally omit customer_ids and rely on default-account fallback.
$preconditions = [ordered]@{
  backend = $Backend
  session_id_present = (-not [string]::IsNullOrWhiteSpace($SessionId))
  target_customer_id = $TargetCustomerId
  effective_customer_id = $null
  effective_account = $null
  oauth_status = $null
  accounts = $null
  default_account_set = $null
  timestamp = (Get-Date).ToString("o")
}
try {
  $statusUrl = "$Backend/api/sa360/oauth/status?session_id=$SessionId"
  $statusResp = Safe-Get $statusUrl
  $preconditions.oauth_status = $statusResp
  $connected = $false
  $currentDefault = $null
  try { $connected = [bool]$statusResp.body.connected } catch {}
  try { $currentDefault = [string]$statusResp.body.default_customer_id } catch {}
  if ($connected) {
    # Prefer an advertiser (non-manager) account for specs that validate metrics. Manager/MCC IDs
    # are valid inputs for account discovery but will correctly fail for metrics endpoints.
    $accountsUrl = "$Backend/api/sa360/accounts?session_id=$([uri]::EscapeDataString($SessionId))"
    $accountsResp = Safe-Get $accountsUrl
    $preconditions.accounts = $accountsResp

    $accounts = @()
    try {
      if ($accountsResp.ok -and $accountsResp.body) { $accounts = @($accountsResp.body) }
    } catch { $accounts = @() }

    $effective = $null
    if ($accounts.Count -gt 0) {
      if ($TargetCustomerId) {
        $requested = ($accounts | Where-Object { $_.customer_id -eq $TargetCustomerId } | Select-Object -First 1)
        # If the caller provided an MCC/manager ID, keep it for traceability but do not use it
        # as the effective metrics target (SA360 metrics endpoints require a child account).
        if ($requested -and $requested.manager -ne $true) { $effective = $requested }
      }
      if (-not $effective) {
        $effective = ($accounts | Where-Object { $_.manager -ne $true } | Select-Object -First 1)
        if (-not $effective -and $requested) { $effective = $requested }
        if (-not $effective) { $effective = $accounts | Select-Object -First 1 }
      }
    }

    if ($effective) {
      $effectiveId = [string]$effective.customer_id
      $preconditions.effective_customer_id = $effectiveId
      $preconditions.effective_account = [ordered]@{
        customer_id = $effectiveId
        name = $effective.name
        manager = $effective.manager
      }

      if (-not $currentDefault -or $currentDefault -ne $effectiveId) {
        $setUrl = "$Backend/api/sa360/default-account"
        $setBody = [ordered]@{
          session_id = $SessionId
          customer_id = $effectiveId
          account_name = $effective.name
        }
        $setResp = Safe-Post $setUrl $setBody
        $preconditions.default_account_set = $setResp
      }
    }
  }
} catch {
  $preconditions.default_account_set = [ordered]@{
    ok = $false
    error = $_.Exception.Message
  }
}
Write-Json (Join-Path $OutDir "preconditions.json") $preconditions

$effectiveTargetForSpecs = $TargetCustomerId
try {
  if ($preconditions.effective_customer_id) { $effectiveTargetForSpecs = [string]$preconditions.effective_customer_id }
} catch {}
$vars = @{
  "SESSION_ID" = $SessionId
  "TARGET_CUSTOMER_ID" = $effectiveTargetForSpecs
}

# Some SA360 metrics can drift minute-to-minute when the range includes very recent days. We observed
# parity deltas > tolerance even when excluding "today" (i.e., including "yesterday"), so for
# deterministic parity/accuracy checks we use a stable "last 7 complete days" window that ends 2 days
# ago to reduce the chance of intraday/backfill drift between the plan call and diagnostics snapshot.
$stableEnd = (Get-Date).Date.AddDays(-2)
$stableStart = $stableEnd.AddDays(-6)
$vars["STABLE_LAST_7_DAYS"] = ("{0:yyyy-MM-dd},{1:yyyy-MM-dd}" -f $stableStart, $stableEnd)
$stableStart14 = $stableEnd.AddDays(-13)
$vars["STABLE_LAST_14_DAYS"] = ("{0:yyyy-MM-dd},{1:yyyy-MM-dd}" -f $stableStart14, $stableEnd)
$stableStart30 = $stableEnd.AddDays(-29)
$vars["STABLE_LAST_30_DAYS"] = ("{0:yyyy-MM-dd},{1:yyyy-MM-dd}" -f $stableStart30, $stableEnd)

$specFiles = Get-ChildItem -Path $SpecDir -Filter *.json -Recurse | Sort-Object FullName
if ($specFiles.Count -eq 0) {
  throw "No spec files found in: $SpecDir"
}

$results = @()
foreach ($f in $specFiles) {
  $raw = Get-Content $f.FullName -Raw
  $spec = $raw | ConvertFrom-Json
  $validationErrors = Validate-Spec $spec $f.Name
  if ($validationErrors.Count -gt 0) {
    $specId = [string]$spec.id
    if ([string]::IsNullOrWhiteSpace($specId)) { $specId = [string]$f.BaseName }
    # Ensure invalid specs still carry capability tags so summary aggregation and CI gates
    # don't crash on null hashtable keys.
    $pillarTag = [string]$spec.pillar
    if ([string]::IsNullOrWhiteSpace($pillarTag)) { $pillarTag = "unknown" }
    $kindTag = [string]$spec.kind
    if ([string]::IsNullOrWhiteSpace($kindTag)) { $kindTag = "unknown" }
    $invalidCaps = @("pillar:$pillarTag", "kind:$kindTag", "error:spec_validation") |
      ForEach-Object { ([string]$_).Trim().ToLower() } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
      Select-Object -Unique
    $results += [ordered]@{
      id = $specId
      scenario = [string]$spec.scenario
      pillar = [string]$spec.pillar
      kind = [string]$spec.kind
      capabilities = $invalidCaps
      ok = $false
      latency_ms = 0
      checks = @()
      assertions = [ordered]@{}
      evidence = [ordered]@{ total = 1; passed = 0 }
      failures = @(
        [ordered]@{ type = "spec_validation"; details = $validationErrors }
      )
      spec_file = $f.Name
    }
    continue
  }
  $specId = [string]$spec.id
  $scenario = [string]$spec.scenario
  $pillar = [string]$spec.pillar
  $kind = [string]$spec.kind

  $request = Substitute $spec.request $vars
  # Allow variable substitution in expectations/assertions so specs can stay portable
  # across environments (e.g., different default customer IDs per QA run).
  $expect = Substitute $spec.expect $vars
  $assert = Substitute $spec.assert $vars
  $intentGoal = Substitute $spec.intent_goal $vars
  $budgets = Substitute $spec.budgets $vars
  $capabilityTags = Get-CapabilityTags $spec $pillar $kind

  $trace = [ordered]@{
    id = $specId
    pillar = $pillar
    kind = $kind
    request = $request
    baseline = $null
    metamorphic = @()
    followup = $null
  }

  $endpoint = ""
  if ($kind -eq "route") {
    $endpoint = "$Backend/api/chat/route"
    $resp = Safe-Post $endpoint $request
  } elseif ($kind -eq "plan_and_run") {
    $endpoint = "$Backend/api/chat/plan-and-run"
    $resp = Safe-Post $endpoint $request
  } elseif ($kind -eq "advisor_diag") {
    $endpoint = "$Backend/api/diagnostics/advisor"
    $resp = Safe-Post $endpoint $request
  } elseif ($kind -eq "send") {
    $endpoint = "$Backend/api/chat/send"
    $resp = Safe-Post $endpoint $request
  } elseif ($kind -eq "get") {
    $url = $null
    try { $url = [string]$request.url } catch {}
    if ([string]::IsNullOrWhiteSpace($url)) {
      try { $url = [string]$request.path } catch {}
    }
    if ([string]::IsNullOrWhiteSpace($url)) {
      throw "GET spec '$specId' missing request.url (or request.path)"
    }
    if ($url -match '^https?://') { $endpoint = $url } else { $endpoint = "$Backend$url" }
    $resp = Safe-Get $endpoint
  } else {
    throw "Unsupported spec kind '$kind' in $($f.Name)"
  }
  $trace.baseline = $resp

  $checks = @()
  $ok = $true
  foreach ($p in $expect.PSObject.Properties) {
    $path = [string]$p.Name
    $expected = $p.Value
    $actual = Get-AnyPathValue $resp $path
    $pass = $true
    if ($expected -is [bool]) {
      $pass = ([bool]$actual -eq [bool]$expected)
    } elseif ($expected -is [double] -or $expected -is [int]) {
      $pass = ($actual -ne $null -and [double]$actual -eq [double]$expected)
    } else {
      $pass = ([string]$actual -eq [string]$expected)
    }
    $checks += [ordered]@{ path = $path; expected = $expected; actual = $actual; ok = $pass }
    if (-not $pass) { $ok = $false }
  }

  # Extra assertions
  $assertions = [ordered]@{}
  # Budget assertions (performance gate).
  if ($budgets -and $budgets.max_latency_ms) {
    $max = [double]$budgets.max_latency_ms
    $pass = ($resp.latency_ms -le $max)
    $assertions.budget_latency = [ordered]@{ max_latency_ms = $max; latency_ms = $resp.latency_ms; ok = $pass }
    if (-not $pass) { $ok = $false }
  }
  if ($budgets -and $budgets.p95_latency_ms) {
    $max = [double]$budgets.p95_latency_ms
    $samples = 5
    try {
      if ($budgets.samples) { $samples = [int]$budgets.samples }
    } catch {}
    if ($samples -lt 2) { $samples = 2 }

    $lat = @()
    $sampleResponses = @()
    for ($i = 0; $i -lt $samples; $i++) {
      $r = Safe-Post $endpoint $request
      $lat += [double]$r.latency_ms
      $sampleResponses += $r
    }
    $p95 = Pctl $lat 0.95
    $pass = ($p95 -ne $null -and $p95 -le $max)
    $assertions.budget_p95_latency = [ordered]@{
      samples = $samples
      max_p95_latency_ms = $max
      p95_latency_ms = $p95
      min_latency_ms = ($lat | Measure-Object -Minimum).Minimum
      max_latency_ms = ($lat | Measure-Object -Maximum).Maximum
      ok = $pass
    }
    # Store raw samples in trace (share-safe: latencies only).
    $trace.p95_latency_samples = $lat
    if (-not $pass) { $ok = $false }
  }

  # Schema assertions (functionality gate).
  $schemaOk = $true
  $schemaMissing = @()
  if ($kind -eq "route") {
    foreach ($k in @("intent","run_planner","needs_ids")) {
      if (-not $resp.body -or -not $resp.body.PSObject.Properties[$k]) { $schemaOk = $false; $schemaMissing += $k }
    }
  } elseif ($kind -eq "plan_and_run") {
    foreach ($k in @("executed")) {
      if (-not $resp.body -or -not $resp.body.PSObject.Properties[$k]) { $schemaOk = $false; $schemaMissing += $k }
    }
    try {
      if ($resp.body.executed -eq $true) {
        if (-not $resp.body.PSObject.Properties["result"]) { $schemaOk = $false; $schemaMissing += "result" }
      }
    } catch {}
  } elseif ($kind -eq "advisor_diag") {
    foreach ($k in @("status","pass")) {
      if (-not $resp.body -or -not $resp.body.PSObject.Properties[$k]) { $schemaOk = $false; $schemaMissing += $k }
    }
  } elseif ($kind -eq "send") {
    foreach ($k in @("reply","model")) {
      if (-not $resp.body -or -not $resp.body.PSObject.Properties[$k]) { $schemaOk = $false; $schemaMissing += $k }
    }
  } elseif ($kind -eq "get") {
    # GET specs are arbitrary; schema checks are opt-in via assert.paths_exist.
  }
  $assertions.schema_valid = [ordered]@{ ok = $schemaOk; missing = ($schemaMissing | Select-Object -Unique) }
  if (-not $schemaOk) { $ok = $false }

  if ($assert -and $assert.summary_matches_result) {
    $assertions.summary_matches_result = Assert-SummaryMatchesResult $resp.body $assert.summary_matches_result
    if (-not $assertions.summary_matches_result.ok) { $ok = $false }
  }
  if ($assert -and $assert.sa360_parity) {
    $assertions.sa360_parity = Assert-Sa360Parity $resp $request $assert.sa360_parity $Backend $SessionId
    if (-not $assertions.sa360_parity.ok) { $ok = $false }
  }
  if ($assert -and $assert.sa360_accuracy) {
    $assertions.sa360_accuracy = Assert-Sa360Accuracy $resp $request $assert.sa360_accuracy $Backend $SessionId
    if (-not $assertions.sa360_accuracy.ok) { $ok = $false }
  }
  if ($assert -and $assert.paths_exist) {
    $missing = @()
    foreach ($p in @($assert.paths_exist)) {
      $val = Get-AnyPathValue $resp ([string]$p)
      $exists = $true
      if ($val -eq $null) { $exists = $false }
      elseif ($val -is [string] -and [string]::IsNullOrWhiteSpace($val)) { $exists = $false }
      elseif (($val -is [System.Array] -or $val -is [System.Collections.IList]) -and @($val).Count -eq 0) { $exists = $false }
      if (-not $exists) { $missing += [string]$p }
    }
    $assertions.paths_exist = [ordered]@{ ok = (($missing | Measure-Object).Count -eq 0); missing = $missing }
    if (-not $assertions.paths_exist.ok) { $ok = $false }
  }
  if ($assert -and $assert.array_min_len) {
    $checksArr = @()
    $arrOk = $true
    foreach ($p in $assert.array_min_len.PSObject.Properties) {
      $path = [string]$p.Name
      $min = [int]$p.Value
      $val = $null
      if ($path -eq "$") { $val = $resp.body } else { $val = Get-AnyPathValue $resp $path }
      $count = 0
      try {
        if ($val -is [string] -or $val -eq $null) {
          $count = 0
        } else {
          $count = @($val).Count
        }
      } catch { $count = 0 }
      $pass = ($count -ge $min)
      $checksArr += [ordered]@{ path = $path; min = $min; count = $count; ok = $pass }
      if (-not $pass) { $arrOk = $false }
    }
    $assertions.array_min_len = [ordered]@{ ok = $arrOk; checks = $checksArr }
    if (-not $assertions.array_min_len.ok) { $ok = $false }
  }
  if ($assert -and $assert.text_not_contains) {
    $checksText = @()
    $textOk = $true
    foreach ($p in $assert.text_not_contains.PSObject.Properties) {
      $path = [string]$p.Name
      $patterns = @($p.Value)
      $val = [string](Get-AnyPathValue $resp $path)
      $lower = $val.ToLower()
      $hits = @()
      foreach ($pat in $patterns) {
        if (-not [string]::IsNullOrWhiteSpace([string]$pat) -and $lower.Contains(([string]$pat).ToLower())) {
          $hits += [string]$pat
        }
      }
      $pass = (($hits | Measure-Object).Count -eq 0)
      $checksText += [ordered]@{ path = $path; patterns = $patterns; hits = $hits; ok = $pass }
      if (-not $pass) { $textOk = $false }
    }
    $assertions.text_not_contains = [ordered]@{ ok = $textOk; checks = $checksText }
    if (-not $assertions.text_not_contains.ok) { $ok = $false }
  }
  if ($assert -and $assert.text_contains) {
    $checksText = @()
    $textOk = $true
    foreach ($p in $assert.text_contains.PSObject.Properties) {
      $path = [string]$p.Name
      $patterns = @($p.Value)
      $val = [string](Get-AnyPathValue $resp $path)
      $lower = $val.ToLower()
      $missing = @()
      foreach ($pat in $patterns) {
        if (-not [string]::IsNullOrWhiteSpace([string]$pat) -and -not $lower.Contains(([string]$pat).ToLower())) {
          $missing += [string]$pat
        }
      }
      $pass = (($missing | Measure-Object).Count -eq 0)
      $checksText += [ordered]@{ path = $path; patterns = $patterns; missing = $missing; ok = $pass }
      if (-not $pass) { $textOk = $false }
    }
    $assertions.text_contains = [ordered]@{ ok = $textOk; checks = $checksText }
    if (-not $assertions.text_contains.ok) { $ok = $false }
  }
  if ($assert -and $assert.regex_match) {
    $checksRx = @()
    $rxOk = $true
    foreach ($p in $assert.regex_match.PSObject.Properties) {
      $path = [string]$p.Name
      $pattern = [string]$p.Value
      $val = [string](Get-AnyPathValue $resp $path)
      $pass = $false
      try { $pass = [regex]::IsMatch($val, $pattern) } catch { $pass = $false }
      $checksRx += [ordered]@{ path = $path; pattern = $pattern; value = $val; ok = $pass }
      if (-not $pass) { $rxOk = $false }
    }
    $assertions.regex_match = [ordered]@{ ok = $rxOk; checks = $checksRx }
    if (-not $assertions.regex_match.ok) { $ok = $false }
  }
  if ($kind -eq "send" -and $assert) {
    if ($assert.reply_min_len) {
      $min = [int]$assert.reply_min_len
      $reply = ""
      try { if ($resp.body -and $resp.body.reply) { $reply = [string]$resp.body.reply } } catch {}
      $pass = ($reply -and $reply.Length -ge $min)
      $assertions.reply_min_len = [ordered]@{ min = $min; actual = ($reply.Length); ok = $pass }
      if (-not $pass) { $ok = $false }
    }
    if ($assert.reply_not_contains) {
      $reply = ""
      try { if ($resp.body -and $resp.body.reply) { $reply = [string]$resp.body.reply } } catch {}
      $lower = $reply.ToLower()
      $patterns = @($assert.reply_not_contains)
      $hits = @()
      foreach ($p in $patterns) {
        if (-not [string]::IsNullOrWhiteSpace([string]$p) -and $lower.Contains(([string]$p).ToLower())) {
          $hits += [string]$p
        }
      }
      $pass = (($hits | Measure-Object).Count -eq 0)
      $assertions.reply_not_contains = [ordered]@{ patterns = $patterns; hits = $hits; ok = $pass }
      if (-not $pass) { $ok = $false }
    }
    if ($assert.reply_regex_not_match) {
      $reply = ""
      try { if ($resp.body -and $resp.body.reply) { $reply = [string]$resp.body.reply } } catch {}
      $checksRxN = @()
      $rxNOk = $true
      foreach ($p in $assert.reply_regex_not_match.PSObject.Properties) {
        $path = [string]$p.Name
        $pattern = [string]$p.Value
        $val = $reply
        $pass = $true
        try { $pass = -not [regex]::IsMatch($val, $pattern) } catch { $pass = $true }
        $checksRxN += [ordered]@{ path = $path; pattern = $pattern; value = $val; ok = $pass }
        if (-not $pass) { $rxNOk = $false }
      }
      $assertions.reply_regex_not_match = [ordered]@{ ok = $rxNOk; checks = $checksRxN }
      if (-not $assertions.reply_regex_not_match.ok) { $ok = $false }
    }
  }
  if ($assert -and $assert.custom_metric_suggestions_min) {
    $sugs = @()
    try { $sugs = @($resp.body.analysis.custom_metric.suggestions) | Where-Object { $_ } } catch {}
    $min = [int]$assert.custom_metric_suggestions_min
    $pass = (($sugs | Measure-Object).Count -ge $min)
    $assertions.custom_metric_suggestions = [ordered]@{
      min = $min
      count = ($sugs | Measure-Object).Count
      ok = $pass
      sample = ($sugs | Select-Object -First 5)
    }
    if (-not $pass) { $ok = $false }
  }
  if ($assert -and $assert.drivers_min) {
    $camp = 0
    $dev = 0
    try { $camp = @($resp.body.analysis.drivers.campaign).Count } catch {}
    try { $dev = @($resp.body.analysis.drivers.device).Count } catch {}
    $min = [int]$assert.drivers_min
    $pass = (($camp + $dev) -ge $min)
    $assertions.drivers = [ordered]@{ min = $min; campaign = $camp; device = $dev; ok = $pass }
    if (-not $pass) { $ok = $false }
  }

  # Advisor diagnostics: ensure evidence JSON is parseable and contains required keys.
  if ($kind -eq "advisor_diag") {
    $evidenceParsedOk = $true
    $evidenceCount = 0
    $missingKeys = @()
    try {
      $evidenceCount = @($resp.body.evidence).Count
      foreach ($e in @($resp.body.evidence)) {
        $obj = $e | ConvertFrom-Json
        foreach ($k in @("insight","driver","action","expected_impact","risk","confidence")) {
          if (-not $obj.PSObject.Properties[$k]) { $missingKeys += $k; $evidenceParsedOk = $false }
        }
      }
    } catch {
      $evidenceParsedOk = $false
    }
    $assertions.advisor_evidence_schema = [ordered]@{
      evidence_count = $evidenceCount
      ok = ($evidenceParsedOk -and $evidenceCount -ge 1)
      missing_keys = ($missingKeys | Select-Object -Unique)
    }
    if (-not $assertions.advisor_evidence_schema.ok) { $ok = $false }
  }

  # Metamorphic tests (meaning-preserving edits should keep key fields stable).
  if ($spec.metamorphic) {
    $metaTotal = 0
    $metaFailed = 0
    $metaFailures = @()
    foreach ($m in @($spec.metamorphic)) {
      $variantMsg = Substitute $m.message $vars
      $variantReq = [ordered]@{}
      # $request is typically an [ordered] dictionary after substitution; PSObject.Properties on a dictionary
      # enumerates type properties rather than the request keys. Copy via Keys when possible.
      if ($request -is [System.Collections.IDictionary]) {
        foreach ($k in $request.Keys) { $variantReq[$k] = $request[$k] }
      } else {
        foreach ($p in $request.PSObject.Properties) { $variantReq[$p.Name] = $p.Value }
      }
      if ($variantReq.Contains("message")) { $variantReq["message"] = $variantMsg }
      elseif ($variantReq.Contains("text")) { $variantReq["text"] = $variantMsg }
      else { $variantReq["message"] = $variantMsg }
      $vResp = Safe-Post $endpoint $variantReq
      $fields = @($m.expect_same)
      $same = $true
      $fieldChecks = @()
      foreach ($field in $fields) {
        $baseVal = Get-PathValue $resp.body $field
        $varVal = Get-PathValue $vResp.body $field
        $pass = ([string]$baseVal -eq [string]$varVal)
        $fieldChecks += [ordered]@{ field = $field; base = $baseVal; variant = $varVal; ok = $pass }
        if (-not $pass) { $same = $false }
      }
      $metaTotal++
      $trace.metamorphic += [ordered]@{
        name = $m.name
        message = $variantMsg
        ok = $same
        checks = $fieldChecks
        response = $vResp
      }
      if (-not $same) {
        $metaFailed++
        $mismatches = @()
        foreach ($c in $fieldChecks) {
          if (-not $c.ok) {
            $mismatches += [ordered]@{ field = $c.field; base = $c.base; variant = $c.variant }
          }
        }
        $metaFailures += [ordered]@{ name = $m.name; mismatches = $mismatches }
        $ok = $false
      }
    }
    $assertions.metamorphic = [ordered]@{
      total = $metaTotal
      failed = $metaFailed
      ok = ($metaFailed -eq 0)
      failures = $metaFailures
    }
  }

  # Followup polling for queued audit jobs (functionality gate).
  $follow = Substitute $spec.followup $vars
  if ($follow -and $follow.poll_job_result -eq $true) {
    $jobOk = $false
    $jobId = $resp.body.job_id
    $timeoutSec = [int]($follow.job_timeout_sec | ForEach-Object { $_ })
    if (-not $timeoutSec) { $timeoutSec = 180 }
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    $jobStatus = $null
    while ((Get-Date) -lt $deadline) {
      $jobStatus = Safe-Get "$Backend/api/jobs/$jobId"
      $state = $null
      try { $state = $jobStatus.body.job.status } catch {}
      if ($state -eq "succeeded" -or $state -eq "failed") { break }
      Start-Sleep -Seconds 2
    }
    $jobResult = $null
    if ($jobStatus -and $jobStatus.ok) {
      $jobResult = Safe-Get "$Backend/api/jobs/$jobId/result"
      if ($jobResult.ok -and $jobResult.body -and $jobResult.body.status -eq "success") { $jobOk = $true }
    }
    $trace.followup = [ordered]@{
      job_id = $jobId
      ok = $jobOk
      status = $jobStatus
      result = $jobResult
    }
    # Bubble followup failures into the normalized assertions so results.json can be used as a CI gate.
    # Keep this share-safe: include statuses, not full payloads.
    $jobState = $null
    try { $jobState = $jobStatus.body.job.status } catch {}
    $jobResultState = $null
    try { $jobResultState = $jobResult.body.status } catch {}
    $assertions.followup_poll_job = [ordered]@{
      ok = $jobOk
      job_id = $jobId
      job_status = $jobState
      result_status = $jobResultState
    }
    if (-not $jobOk) { $ok = $false }
  }

  # Followup chat send (planner summary / tool followup) to validate multi-turn UX.
  # This is used to assert "advisor-grade" behavior after a tool run, without re-running the planner.
  if ($follow -and $follow.chat_send) {
    $followCfg = $follow.chat_send
    $followMsg = Substitute ([string]$followCfg.message) $vars
    if ([string]::IsNullOrWhiteSpace($followMsg)) {
      $followMsg = "Summarize the planner output."
    }
    $followTool = [string]$followCfg.tool
    if ([string]::IsNullOrWhiteSpace($followTool)) { $followTool = "performance" }
    $followPromptKind = [string]$followCfg.prompt_kind
    if ([string]::IsNullOrWhiteSpace($followPromptKind)) { $followPromptKind = "planner_summary" }

    $followSession = $SessionId
    try {
      if ($request -and $request.session_id) { $followSession = [string]$request.session_id }
    } catch {}

    $followReq = [ordered]@{
      message = $followMsg
      ai_enabled = $true
      session_id = $followSession
      context = [ordered]@{
        tool = $followTool
        tool_output = $resp.body
        prompt_kind = $followPromptKind
      }
    }
    $followEndpoint = "$Backend/api/chat/send"
    $followResp = Safe-Post $followEndpoint $followReq

    $followOk = $followResp.ok
    $followChecks = @()
    $followAssertions = [ordered]@{}

    # Expectations (path/value equality) for followup response.
    $followExpect = $followCfg.expect
    if ($followExpect) {
      foreach ($p in $followExpect.PSObject.Properties) {
        $path = [string]$p.Name
        $expected = $p.Value
        $actual = Get-AnyPathValue $followResp $path
        $pass = $true
        if ($expected -is [bool]) {
          $pass = ([bool]$actual -eq [bool]$expected)
        } elseif ($expected -is [double] -or $expected -is [int]) {
          $pass = ($actual -ne $null -and [double]$actual -eq [double]$expected)
        } else {
          $pass = ([string]$actual -eq [string]$expected)
        }
        $followChecks += [ordered]@{ path = $path; expected = $expected; actual = $actual; ok = $pass }
        if (-not $pass) { $followOk = $false }
      }
    }

    # Followup budget assertions (performance gate).
    $followBudgets = $followCfg.budgets
    if ($followBudgets -and $followBudgets.max_latency_ms) {
      $max = [double]$followBudgets.max_latency_ms
      $pass = ($followResp.latency_ms -le $max)
      $followAssertions.budget_latency = [ordered]@{ max_latency_ms = $max; latency_ms = $followResp.latency_ms; ok = $pass }
      if (-not $pass) { $followOk = $false }
    }

    # Followup assertions: reuse existing primitive checkers (regex_match/text_contains/etc).
    $followAssert = $followCfg.assert
    if ($followAssert) {
      if ($followAssert.model_in) {
        $models = @($followAssert.model_in) | ForEach-Object { [string]$_ }
        $actualModel = ""
        try { if ($followResp.body -and $followResp.body.model) { $actualModel = [string]$followResp.body.model } } catch {}
        $pass = ($models -contains $actualModel)
        $followAssertions.model_in = [ordered]@{ allowed = $models; actual = $actualModel; ok = $pass }
        if (-not $pass) { $followOk = $false }
      }
      if ($followAssert.model_not_in) {
        $models = @($followAssert.model_not_in) | ForEach-Object { [string]$_ }
        $actualModel = ""
        try { if ($followResp.body -and $followResp.body.model) { $actualModel = [string]$followResp.body.model } } catch {}
        $pass = -not ($models -contains $actualModel)
        $followAssertions.model_not_in = [ordered]@{ forbidden = $models; actual = $actualModel; ok = $pass }
        if (-not $pass) { $followOk = $false }
      }
      if ($followAssert.reply_min_len) {
        $min = [int]$followAssert.reply_min_len
        $reply = ""
        try { if ($followResp.body -and $followResp.body.reply) { $reply = [string]$followResp.body.reply } } catch {}
        $pass = ($reply -and $reply.Length -ge $min)
        $followAssertions.reply_min_len = [ordered]@{ min = $min; actual = ($reply.Length); ok = $pass }
        if (-not $pass) { $followOk = $false }
      }
      if ($followAssert.reply_not_contains) {
        $reply = ""
        try { if ($followResp.body -and $followResp.body.reply) { $reply = [string]$followResp.body.reply } } catch {}
        $lower = $reply.ToLower()
        $patterns = @($followAssert.reply_not_contains)
        $hits = @()
        foreach ($p in $patterns) {
          if (-not [string]::IsNullOrWhiteSpace([string]$p) -and $lower.Contains(([string]$p).ToLower())) {
            $hits += [string]$p
          }
        }
        $pass = (($hits | Measure-Object).Count -eq 0)
        $followAssertions.reply_not_contains = [ordered]@{ patterns = $patterns; hits = $hits; ok = $pass }
        if (-not $pass) { $followOk = $false }
      }
      if ($followAssert.reply_regex_not_match) {
        $reply = ""
        try { if ($followResp.body -and $followResp.body.reply) { $reply = [string]$followResp.body.reply } } catch {}
        $checksRxN = @()
        $rxNOk = $true
        foreach ($p in $followAssert.reply_regex_not_match.PSObject.Properties) {
          $path = [string]$p.Name
          $pattern = [string]$p.Value
          $val = $reply
          $pass = $true
          try { $pass = -not [regex]::IsMatch($val, $pattern) } catch { $pass = $true }
          $checksRxN += [ordered]@{ path = $path; pattern = $pattern; value = $val; ok = $pass }
          if (-not $pass) { $rxNOk = $false }
        }
        $followAssertions.reply_regex_not_match = [ordered]@{ ok = $rxNOk; checks = $checksRxN }
        if (-not $followAssertions.reply_regex_not_match.ok) { $followOk = $false }
      }
      if ($followAssert.regex_match) {
        $checksRx = @()
        $rxOk = $true
        foreach ($p in $followAssert.regex_match.PSObject.Properties) {
          $path = [string]$p.Name
          $pattern = [string]$p.Value
          $val = [string](Get-AnyPathValue $followResp $path)
          $pass = $false
          try { $pass = [regex]::IsMatch($val, $pattern) } catch { $pass = $false }
          $checksRx += [ordered]@{ path = $path; pattern = $pattern; value = $val; ok = $pass }
          if (-not $pass) { $rxOk = $false }
        }
        $followAssertions.regex_match = [ordered]@{ ok = $rxOk; checks = $checksRx }
        if (-not $followAssertions.regex_match.ok) { $followOk = $false }
      }
      if ($followAssert.text_contains) {
        $checksText = @()
        $textOk = $true
        foreach ($p in $followAssert.text_contains.PSObject.Properties) {
          $path = [string]$p.Name
          $patterns = @($p.Value)
          $val = [string](Get-AnyPathValue $followResp $path)
          $lower = $val.ToLower()
          $missing = @()
          foreach ($pat in $patterns) {
            if (-not [string]::IsNullOrWhiteSpace([string]$pat) -and -not $lower.Contains(([string]$pat).ToLower())) {
              $missing += [string]$pat
            }
          }
          $pass = (($missing | Measure-Object).Count -eq 0)
          $checksText += [ordered]@{ path = $path; patterns = $patterns; missing = $missing; ok = $pass }
          if (-not $pass) { $textOk = $false }
        }
        $followAssertions.text_contains = [ordered]@{ ok = $textOk; checks = $checksText }
        if (-not $followAssertions.text_contains.ok) { $followOk = $false }
      }
      if ($followAssert.text_not_contains) {
        $checksText = @()
        $textOk = $true
        foreach ($p in $followAssert.text_not_contains.PSObject.Properties) {
          $path = [string]$p.Name
          $patterns = @($p.Value)
          $val = [string](Get-AnyPathValue $followResp $path)
          $lower = $val.ToLower()
          $hits = @()
          foreach ($pat in $patterns) {
            if (-not [string]::IsNullOrWhiteSpace([string]$pat) -and $lower.Contains(([string]$pat).ToLower())) {
              $hits += [string]$pat
            }
          }
          $pass = (($hits | Measure-Object).Count -eq 0)
          $checksText += [ordered]@{ path = $path; patterns = $patterns; hits = $hits; ok = $pass }
          if (-not $pass) { $textOk = $false }
        }
        $followAssertions.text_not_contains = [ordered]@{ ok = $textOk; checks = $checksText }
        if (-not $followAssertions.text_not_contains.ok) { $followOk = $false }
      }
    }

    $trace.followup = [ordered]@{
      kind = "chat_send"
      request = $followReq
      response = $followResp
      ok = $followOk
      checks = $followChecks
      assertions = $followAssertions
    }
    # Bubble followup failures into the normalized assertions so results.json can be used as a CI gate.
    $assertions.followup_chat_send = [ordered]@{
      ok = $followOk
      checks = $followChecks
      assertions = $followAssertions
    }
    if (-not $followOk) { $ok = $false }
  }

  # Intent -> goal alignment assertion (baseline or follow-up response).
  if ($intentGoal) {
    $target = ""
    try { $target = [string]$intentGoal.target } catch {}
    if ([string]::IsNullOrWhiteSpace($target)) { $target = "baseline" }

    $targetResp = $resp
    if ($target -eq "followup_chat_send") {
      try {
        if ($trace.followup -and $trace.followup.kind -eq "chat_send" -and $trace.followup.response) {
          $targetResp = $trace.followup.response
        }
      } catch {}
    }
    $alignment = Evaluate-IntentGoalAlignment $targetResp $intentGoal
    $alignment.target = $target
    $assertions.intent_goal = $alignment
    if (-not $alignment.ok) { $ok = $false }
  }

  Write-Json (Join-Path $TraceDir ($specId + ".json")) $trace

  # Normalize failures into a single field so CI/drift monitors can reason about causes without reading traces.
  $failures = @()
  $badChecks = @($checks | Where-Object { -not $_.ok })
  if ($badChecks.Count -gt 0) {
    $failures += [ordered]@{ type = "expect"; count = $badChecks.Count; details = $badChecks }
  }
  foreach ($k in @($assertions.Keys)) {
    $v = $assertions[$k]
    if ($v -is [System.Collections.IDictionary]) {
      $hasOk = $false
      try { $hasOk = $v.Contains("ok") } catch { $hasOk = $false }
      if ($hasOk -and (-not [bool]$v["ok"])) {
        $failures += [ordered]@{ type = "assert"; key = $k; details = $v }
      }
    }
  }

  $evidence = Get-EvidenceStats $checks $assertions $ok
  $wilson = Get-WilsonLowerBound $evidence.passed $evidence.total
  $confidenceLevel = Get-ConfidenceLevel $wilson $evidence.total

  $results += [ordered]@{
    id = $specId
    scenario = $scenario
    pillar = $pillar
    kind = $kind
    capabilities = $capabilityTags
    ok = $ok
    latency_ms = $resp.latency_ms
    checks = $checks
    assertions = $assertions
    evidence = $evidence
    confidence = [ordered]@{
      method = "wilson_lower_bound_95"
      score = $wilson
      level = $confidenceLevel
    }
    failures = $failures
    spec_file = $f.Name
  }
}

$byPillar = [ordered]@{}
foreach ($p in @("intent","purpose","quality","functionality")) {
  $items = @($results | Where-Object { $_.pillar -eq $p })
  $pass = @($items | Where-Object { $_.ok }).Count
  $total = $items.Count
  $rate = 1.0
  if ($total -gt 0) { $rate = [math]::Round($pass / $total, 4) }
  $byPillar[$p] = [ordered]@{ total = $total; passed = $pass; pass_rate = $rate }
}

$summary = [ordered]@{
  timestamp = (Get-Date).ToString("o")
  backend = $Backend
  spec_dir = $SpecDir
  out_dir = $OutDir
  total = $results.Count
  passed = @($results | Where-Object { $_.ok }).Count
  failed = @($results | Where-Object { -not $_.ok }).Count
  pillars = $byPillar
  intent_goal_alignment = [ordered]@{
    total = @($results | Where-Object { $_.assertions -and $_.assertions.intent_goal }).Count
    passed = @($results | Where-Object { $_.assertions -and $_.assertions.intent_goal -and $_.assertions.intent_goal.ok }).Count
  }
  sa360_parity = [ordered]@{
    total = @($results | Where-Object { $_.assertions -and $_.assertions.sa360_parity }).Count
    passed = @($results | Where-Object { $_.assertions -and $_.assertions.sa360_parity -and $_.assertions.sa360_parity.ok }).Count
  }
  sa360_accuracy = [ordered]@{
    total = @($results | Where-Object { $_.assertions -and $_.assertions.sa360_accuracy }).Count
    passed = @($results | Where-Object { $_.assertions -and $_.assertions.sa360_accuracy -and $_.assertions.sa360_accuracy.ok }).Count
  }
}

$capabilityRaw = @{}
foreach ($r in @($results)) {
  $caps = @()
  try { $caps = @($r.capabilities) } catch { $caps = @() }
  if ($caps.Count -eq 0) {
    $pillarTag = "unknown"
    $kindTag = "unknown"
    try {
      if (-not [string]::IsNullOrWhiteSpace([string]$r.pillar)) { $pillarTag = [string]$r.pillar }
      if (-not [string]::IsNullOrWhiteSpace([string]$r.kind)) { $kindTag = [string]$r.kind }
    } catch {}
    $caps = @("pillar:$pillarTag", "kind:$kindTag", "error:missing_capabilities")
  }

  foreach ($capRaw in $caps) {
    $cap = ([string]$capRaw).Trim().ToLower()
    if ([string]::IsNullOrWhiteSpace($cap)) { continue }

    if (-not $capabilityRaw.ContainsKey($cap)) {
      $capabilityRaw[$cap] = [ordered]@{
        specs_total = 0
        specs_passed = 0
        evidence_total = 0
        evidence_passed = 0
      }
    }
    $row = $capabilityRaw[$cap]
    $row.specs_total += 1
    if ($r.ok) { $row.specs_passed += 1 }
    try {
      $row.evidence_total += [int]$r.evidence.total
      $row.evidence_passed += [int]$r.evidence.passed
    } catch {}
  }
}

$capabilityConfidence = [ordered]@{}
foreach ($cap in ($capabilityRaw.Keys | Sort-Object)) {
  $row = $capabilityRaw[$cap]
  $eTotal = [int]$row.evidence_total
  $ePassed = [int]$row.evidence_passed
  $passRate = 0.0
  if ($eTotal -gt 0) { $passRate = [math]::Round(($ePassed / $eTotal), 4) }
  $wilson = Get-WilsonLowerBound $ePassed $eTotal
  $level = Get-ConfidenceLevel $wilson $eTotal
  $capabilityConfidence[$cap] = [ordered]@{
    specs_total = [int]$row.specs_total
    specs_passed = [int]$row.specs_passed
    evidence_total = $eTotal
    evidence_passed = $ePassed
    pass_rate = $passRate
    confidence = [ordered]@{
      method = "wilson_lower_bound_95"
      score = $wilson
      level = $level
    }
  }
}
$summary.capability_confidence = $capabilityConfidence

Write-Json (Join-Path $OutDir "results.json") $results
Write-Json (Join-Path $OutDir "summary.json") $summary
Write-Output "Spec eval folder: $OutDir"

if ($Strict -and $summary.failed -gt 0) {
  exit 1
}
