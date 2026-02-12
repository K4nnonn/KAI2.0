Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$evidenceDir = Join-Path $repoRoot ("verification_runs/license_matrix_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $evidenceDir | Out-Null

function Set-CaseEnv {
    param([hashtable]$Values)
    foreach ($k in $Values.Keys) {
        [Environment]::SetEnvironmentVariable($k, [string]$Values[$k], "Process")
    }
}

function Wait-Health {
    param([int]$TimeoutSeconds = 40)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 3
            if ($resp.status -eq "healthy") { return $true }
        } catch {}
        Start-Sleep -Milliseconds 700
    }
    return $false
}

function Invoke-Case {
    param(
        [string]$Name,
        [hashtable]$EnvValues,
        [scriptblock]$Assertions
    )
    Write-Host "Running case: $Name"
    Set-CaseEnv -Values $EnvValues
    $proc = Start-Process -FilePath python -ArgumentList "services/api/main.py" -WorkingDirectory $repoRoot -PassThru
    try {
        if (-not (Wait-Health -TimeoutSeconds 55)) {
            throw "startup/health failed for case $Name"
        }
        & $Assertions
    } finally {
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
        Start-Sleep -Milliseconds 500
    }
}

$results = [System.Collections.Generic.List[object]]::new()

function Add-Result {
    param([string]$Test,[string]$Expected,[string]$Actual,[bool]$Pass)
    $results.Add([pscustomobject]@{
        test = $Test
        expected = $Expected
        actual = $Actual
        pass = $Pass
    })
}

# Baseline evidence
$gitStatus = git status --short --branch
$gitStatus | Set-Content -Path (Join-Path $evidenceDir "git_status.txt") -Encoding UTF8

Invoke-Case -Name "baseline_off" -EnvValues @{
    LICENSE_SERVER_URL = "http://127.0.0.1:9"
    LICENSE_INSTANCE_ID = "baseline-instance"
    LICENSE_ENFORCEMENT_MODE = "off"
    LICENSE_REFRESH_INTERVAL_HOURS = "24"
    LICENSE_RENEW_DAYS_BEFORE_EXP = "3"
    LICENSE_GRACE_DAYS = "1"
    LICENSE_REQUEST_TIMEOUT_SECONDS = "2"
    LICENSE_CACHE_PATH = (Join-Path $repoRoot "runtime/license_baseline_cache.json")
    BRAIN_GATE_USE_OBFUSCATED = "false"
} -Assertions {
    $health = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 15
    Add-Result -Test "baseline_health" -Expected "200 healthy" -Actual "$($health.StatusCode) $($health.Content)" -Pass ($health.StatusCode -eq 200)

    $diag = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/diagnostics/health" -UseBasicParsing -TimeoutSec 15
    Add-Result -Test "baseline_diag_health" -Expected "200 diagnostics" -Actual "$($diag.StatusCode)" -Pass ($diag.StatusCode -eq 200)

    $chatBody = @{message="baseline chat";session_id="baseline-session"} | ConvertTo-Json
    $chat = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/chat/send" -Method Post -Body $chatBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 60
    Add-Result -Test "baseline_chat_send" -Expected "200 response" -Actual "$($chat.StatusCode)" -Pass ($chat.StatusCode -eq 200)

    $audit = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/audit/business-units" -UseBasicParsing -TimeoutSec 15
    Add-Result -Test "baseline_audit_sanity" -Expected "200 business units" -Actual "$($audit.StatusCode)" -Pass ($audit.StatusCode -eq 200)
}

# Case 2: soft + invalid URL => protected 200 + warning
Invoke-Case -Name "soft_invalid" -EnvValues @{
    LICENSE_SERVER_URL = "http://127.0.0.1:9"
    LICENSE_INSTANCE_ID = "soft-instance"
    LICENSE_ENFORCEMENT_MODE = "soft"
    LICENSE_REFRESH_INTERVAL_HOURS = "24"
    LICENSE_RENEW_DAYS_BEFORE_EXP = "3"
    LICENSE_GRACE_DAYS = "1"
    LICENSE_REQUEST_TIMEOUT_SECONDS = "2"
    LICENSE_CACHE_PATH = (Join-Path $repoRoot "runtime/license_soft_cache.json")
    BRAIN_GATE_USE_OBFUSCATED = "false"
} -Assertions {
    $chatBody = @{message="soft mode test";session_id="soft-session"} | ConvertTo-Json
    $chat = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/chat/send" -Method Post -Body $chatBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 60
    $warn = $chat.Headers["X-License-Warn"]
    Add-Result -Test "soft_invalid_chat_status" -Expected "200 with X-License-Warn" -Actual "status=$($chat.StatusCode);warn=$warn" -Pass (($chat.StatusCode -eq 200) -and -not [string]::IsNullOrWhiteSpace($warn))
}

# Case 3: hard + invalid URL + no cache => protected 503, exempt 200
Invoke-Case -Name "hard_invalid_no_cache" -EnvValues @{
    LICENSE_SERVER_URL = "http://127.0.0.1:9"
    LICENSE_INSTANCE_ID = "hard-instance"
    LICENSE_ENFORCEMENT_MODE = "hard"
    LICENSE_REFRESH_INTERVAL_HOURS = "24"
    LICENSE_RENEW_DAYS_BEFORE_EXP = "3"
    LICENSE_GRACE_DAYS = "1"
    LICENSE_REQUEST_TIMEOUT_SECONDS = "2"
    LICENSE_CACHE_PATH = (Join-Path $repoRoot "runtime/license_hard_no_cache.json")
    BRAIN_GATE_USE_OBFUSCATED = "false"
} -Assertions {
    Remove-Item -Path (Join-Path $repoRoot "runtime/license_hard_no_cache.json") -Force -ErrorAction SilentlyContinue
    $chatBody = @{message="hard mode test";session_id="hard-session"} | ConvertTo-Json
    $blocked = $null
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/chat/send" -Method Post -Body $chatBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 30 | Out-Null
        Add-Result -Test "hard_invalid_protected" -Expected "503 blocked" -Actual "unexpected 200" -Pass $false
    } catch {
        $resp = $_.Exception.Response
        if ($resp -and $resp.StatusCode.value__ -eq 503) {
            Add-Result -Test "hard_invalid_protected" -Expected "503 blocked" -Actual "503 blocked" -Pass $true
        } else {
            Add-Result -Test "hard_invalid_protected" -Expected "503 blocked" -Actual "$($_.Exception.Message)" -Pass $false
        }
    }
    $health = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 10
    Add-Result -Test "hard_invalid_exempt_health" -Expected "200 health still available" -Actual "$($health.StatusCode)" -Pass ($health.StatusCode -eq 200)
}

# Case 4: hard + valid cache + outage => protected 200 within grace
$cachePath = Join-Path $repoRoot "runtime/license_hard_cache.json"
$runtimeDir = Split-Path $cachePath -Parent
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$expiredWithinGrace = [int][double]::Parse((Get-Date).ToUniversalTime().AddHours(-10).Subtract([datetime]'1970-01-01').TotalSeconds.ToString("0"))
@{
    instance_id = "hard-cache-instance"
    sub = "hard-cache-instance"
    exp = $expiredWithinGrace
    exp_iso = (Get-Date).ToUniversalTime().AddHours(-10).ToString("o")
    validated_at = (Get-Date).ToUniversalTime().ToString("o")
} | ConvertTo-Json | Set-Content -Path $cachePath -Encoding ASCII

Invoke-Case -Name "hard_cached_grace" -EnvValues @{
    LICENSE_SERVER_URL = "http://127.0.0.1:9"
    LICENSE_INSTANCE_ID = "hard-cache-instance"
    LICENSE_ENFORCEMENT_MODE = "hard"
    LICENSE_REFRESH_INTERVAL_HOURS = "24"
    LICENSE_RENEW_DAYS_BEFORE_EXP = "3"
    LICENSE_GRACE_DAYS = "1"
    LICENSE_REQUEST_TIMEOUT_SECONDS = "2"
    LICENSE_CACHE_PATH = $cachePath
    BRAIN_GATE_USE_OBFUSCATED = "false"
} -Assertions {
    $chatBody = @{message="hard cached grace test";session_id="hard-cache-session"} | ConvertTo-Json
    $chat = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/chat/send" -Method Post -Body $chatBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 60
    Add-Result -Test "hard_cached_grace_chat" -Expected "200 allowed by grace cache" -Actual "$($chat.StatusCode)" -Pass ($chat.StatusCode -eq 200)

    $lic = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/diagnostics/license" -TimeoutSec 10
    Add-Result -Test "license_diag_fields" -Expected "mode,valid,reason,exp,cache_present,last_check,next_check" -Actual ($lic | ConvertTo-Json -Compress) -Pass (
        $lic.PSObject.Properties.Name -contains "mode" -and
        $lic.PSObject.Properties.Name -contains "valid" -and
        $lic.PSObject.Properties.Name -contains "reason" -and
        $lic.PSObject.Properties.Name -contains "exp" -and
        $lic.PSObject.Properties.Name -contains "cache_present" -and
        $lic.PSObject.Properties.Name -contains "last_check" -and
        $lic.PSObject.Properties.Name -contains "next_check"
    )
}

# Env exposure checks (run in off mode instance)
Invoke-Case -Name "env_exposure_checks" -EnvValues @{
    LICENSE_SERVER_URL = "http://127.0.0.1:9"
    LICENSE_INSTANCE_ID = "env-check"
    LICENSE_ENFORCEMENT_MODE = "off"
    LICENSE_CACHE_PATH = (Join-Path $repoRoot "runtime/license_env_check.json")
    BRAIN_GATE_USE_OBFUSCATED = "false"
    KAI_ACCESS_PASSWORD = "kelvinkai"
    KAI_ENV_GUI_PASSWORD = "kelvinkai"
} -Assertions {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/settings/env?admin_password=kelvinkai" -TimeoutSec 15
    $keys = @($resp.env | ForEach-Object { $_.key })
    $forbidden = @("KEY","SECRET","TOKEN","PASSWORD","CONNECTION_STRING","CLIENT_SECRET","REFRESH_TOKEN")
    $hit = $false
    foreach ($k in $keys) {
        $up = $k.ToUpperInvariant()
        foreach ($p in $forbidden) {
            if ($up.Contains($p)) {
                $hit = $true
            }
        }
    }
    Add-Result -Test "env_no_forbidden_keys" -Expected "No forbidden env key patterns in /api/settings/env keys" -Actual ("keys=" + ($keys -join ",")) -Pass (-not $hit)
}

$resultsPath = Join-Path $evidenceDir "results.json"
$results | ConvertTo-Json -Depth 6 | Set-Content -Path $resultsPath -Encoding UTF8

$passCount = ($results | Where-Object { $_.pass }).Count
$totalCount = $results.Count
Write-Host "Results: $resultsPath"
Write-Host "Passed: $passCount / $totalCount"
if ($passCount -ne $totalCount) {
    throw "License validation matrix failed. See $resultsPath"
}
