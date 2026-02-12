Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptStart = Get-Date
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$runRoot = Join-Path $repoRoot "integrity_runs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $runRoot $timestamp
New-Item -ItemType Directory -Force $runDir | Out-Null

$results = @()

function Invoke-Gate {
    param(
        [string]$Name,
        [string]$Command,
        [string]$WorkDir,
        [int]$Order
    )

    $gateDir = Join-Path $runDir ("gate_{0:D2}_{1}" -f $Order, $Name)
    New-Item -ItemType Directory -Force $gateDir | Out-Null

    $stdoutFile = Join-Path $gateDir "stdout.txt"
    $stderrFile = Join-Path $gateDir "stderr.txt"
    $start = Get-Date
    $exitCode = 0

    $stdout = ""
    $stderr = ""
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "cmd.exe"
        $psi.Arguments = "/c $Command"
        $psi.WorkingDirectory = $WorkDir
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $proc = [System.Diagnostics.Process]::Start($psi)
        $stdout = $proc.StandardOutput.ReadToEnd()
        $stderr = $proc.StandardError.ReadToEnd()
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
    } catch {
        $exitCode = 1
        $stderr += $_.ToString()
    }
    $stdout | Set-Content -Path $stdoutFile -Encoding ascii
    $stderr | Set-Content -Path $stderrFile -Encoding ascii

    $end = Get-Date
    $result = [ordered]@{
        name = $Name
        command = $Command
        workdir = $WorkDir
        start = $start.ToString("o")
        end = $end.ToString("o")
        exit_code = $exitCode
        stdout = $stdoutFile
        stderr = $stderrFile
    }
    $script:results += $result
    $resultsPath = Join-Path $runDir "results.json"
    $script:results | ConvertTo-Json -Depth 6 | Set-Content -Path $resultsPath -Encoding ascii

    if ($exitCode -ne 0) {
        Write-Host "Gate failed: $Name (exit $exitCode)"
        Write-Host "Results: $resultsPath"
        exit $exitCode
    }
}

$web = Join-Path $repoRoot "apps\\web"
if (-not (Test-Path $web)) {
    throw "apps\\web not found at $web"
}
$e2e = Join-Path $repoRoot "tests\\e2e"
if (-not (Test-Path $e2e)) {
    Write-Host "tests\\e2e not found at $e2e - e2e gates will be skipped."
}

$requireEnv = $false
if ($env:REQUIRE_ENV) {
    $requireEnv = $env:REQUIRE_ENV.ToLower() -eq "true"
}
$missingEnv = @()
foreach ($name in @("FRONTEND_URL","BACKEND_URL","KAI_ACCESS_PASSWORD")) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if (-not $value) { $missingEnv += $name }
}
$skipE2E = $false
if ($missingEnv.Count -gt 0) {
    if ($requireEnv) {
        throw ("Missing required env vars: " + ($missingEnv -join ", "))
    } else {
        Write-Host ("Missing env vars (" + ($missingEnv -join ", ") + ") - skipping e2e gates.")
        $skipE2E = $true
    }
}

function Add-HashIfExists {
    param(
        [hashtable]$Table,
        [string]$Key,
        [string]$Path
    )
    if (Test-Path $Path) {
        $Table[$Key] = (Get-FileHash $Path -Algorithm SHA256).Hash
    } else {
        $Table[$Key] = $null
    }
}

$scriptHashes = [ordered]@{}
Add-HashIfExists -Table $scriptHashes -Key "tests\\e2e\\ui_smoke_gate.js" -Path (Join-Path $repoRoot "tests\\e2e\\ui_smoke_gate.js")
Add-HashIfExists -Table $scriptHashes -Key "tests\\e2e\\integrity_suite.js" -Path (Join-Path $repoRoot "tests\\e2e\\integrity_suite.js")
Add-HashIfExists -Table $scriptHashes -Key "tests\\e2e\\health_gate.js" -Path (Join-Path $repoRoot "tests\\e2e\\health_gate.js")
Add-HashIfExists -Table $scriptHashes -Key "tests\\e2e\\conversation_flow_test.js" -Path (Join-Path $repoRoot "tests\\e2e\\conversation_flow_test.js")
Add-HashIfExists -Table $scriptHashes -Key "scripts\\run_build_and_integrity.ps1" -Path $PSCommandPath

$requireVersionMatch = $false
if ($env:REQUIRE_VERSION_MATCH) {
    $requireVersionMatch = $env:REQUIRE_VERSION_MATCH.ToLower() -eq "true"
}
if ($requireVersionMatch) {
    if (-not $env:EXPECTED_FRONTEND_BUILD_SHA -or -not $env:EXPECTED_BACKEND_BUILD_SHA) {
        throw "REQUIRE_VERSION_MATCH=true requires EXPECTED_FRONTEND_BUILD_SHA and EXPECTED_BACKEND_BUILD_SHA"
    }
}

$fullIntegrity = $false
if ($env:FULL_INTEGRITY) {
    $fullIntegrity = $env:FULL_INTEGRITY.ToLower() -eq "true"
}
if ($fullIntegrity) {
    if (-not $env:TEST_CUSTOMER_IDS) {
        throw "FULL_INTEGRITY=true requires TEST_CUSTOMER_IDS to avoid skips in performance gates"
    }
    if (-not (Test-Path $e2e)) {
        throw "FULL_INTEGRITY=true requires tests\\e2e to be present"
    }
    $env:REQUIRE_NO_SKIPS = "true"
    $env:REQUIRE_TEST_CUSTOMER_IDS = "true"
}

Invoke-Gate -Name "build_frontend" -Command "npm run build" -WorkDir $web -Order 1
if ((Test-Path $e2e) -and -not $skipE2E) {
    Invoke-Gate -Name "api_health" -Command "node .\\health_gate.js" -WorkDir $e2e -Order 2
    Invoke-Gate -Name "ui_smoke" -Command "node .\\ui_smoke_gate.js" -WorkDir $e2e -Order 3
    Invoke-Gate -Name "conversation_flow" -Command "set MAX_MESSAGES=9 && node .\\conversation_flow_test.js" -WorkDir $e2e -Order 4
    Invoke-Gate -Name "integrity_suite" -Command "node .\\integrity_suite.js" -WorkDir $e2e -Order 5
} else {
    Write-Host "Skipping e2e gates: tests\\e2e not present."
}

$scriptEnd = Get-Date
$meta = [ordered]@{
    start = $scriptStart.ToString("o")
    end = $scriptEnd.ToString("o")
    repo_root = $repoRoot
    run_dir = $runDir
    exit_code = 0
    script_hashes = $scriptHashes
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $runDir "run_meta.json") -Encoding ascii

Write-Host "Run folder: $runDir"
exit 0
