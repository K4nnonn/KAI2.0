Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "[1/5] Build obfuscated brain_gate"
powershell -ExecutionPolicy Bypass -File .\scripts\build_brain_gate_obfuscated.ps1
if ($LASTEXITCODE -ne 0) {
    throw "FAIL: build_brain_gate_obfuscated.ps1 failed"
}

$obfDir = Join-Path $repoRoot "build/obf"
$artifact = Get-ChildItem -Path $obfDir -File | Where-Object {
    $_.Name -like "brain_gate*.pyd" -or $_.Name -like "brain_gate*.so"
} | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $artifact) {
    throw "FAIL: brain_gate artifact missing in $obfDir"
}
Write-Host "PASS: artifact=$($artifact.Name)"

Write-Host "[2/5] Verify manifest"
$manifest = Join-Path $obfDir "brain_gate_build_manifest.txt"
if (-not (Test-Path $manifest)) {
    throw "FAIL: manifest missing: $manifest"
}
$manifestText = Get-Content $manifest -Raw
if ($manifestText -notmatch "artifact_sha256=") {
    throw "FAIL: manifest missing SHA256 entry"
}
Write-Host "PASS: manifest with SHA256 found"

Write-Host "[3/5] Verify BRAIN_GATE_USE_OBFUSCATED=true works"
$tmp = Join-Path $repoRoot "runtime/license_cache_obf_test.json"
if (Test-Path $tmp) { Remove-Item $tmp -Force }
$cmdOk = @'
import os, sys
sys.path.append("services/api")
os.environ["BRAIN_GATE_USE_OBFUSCATED"]="true"
os.environ["LICENSE_ENFORCEMENT_MODE"]="off"
os.environ["LICENSE_CACHE_PATH"]="runtime/license_cache_obf_test.json"
import main
print(main._brain_gate_status().get("mode"))
'@
$tmpScriptOk = Join-Path $repoRoot "build/obf/verify_obf_ok.py"
$cmdOk | Set-Content -Path $tmpScriptOk -Encoding UTF8
$output = python $tmpScriptOk
if ($LASTEXITCODE -ne 0) { throw "FAIL: obfuscated mode did not load successfully" }
if (-not ($output -match "off")) { throw "FAIL: obfuscated mode output unexpected: $output" }
Write-Host "PASS: obfuscated mode load works"

Write-Host "[4/5] Verify missing artifact + toggle=true fails with controlled error"
$backupDir = Join-Path $repoRoot "build/obf_bak_tmp"
if (Test-Path $backupDir) { Remove-Item $backupDir -Recurse -Force }
Move-Item -Path $obfDir -Destination $backupDir
try {
    $cmdFail = @'
import os, sys
sys.path.append("services/api")
os.environ["BRAIN_GATE_USE_OBFUSCATED"]="true"
os.environ["LICENSE_ENFORCEMENT_MODE"]="off"
import main
main._brain_gate_status()
'@
    $tmpScriptFail = Join-Path $repoRoot "build/obf_bak_tmp/verify_obf_fail.py"
    $cmdFail | Set-Content -Path $tmpScriptFail -Encoding UTF8
    cmd /c "python `"$tmpScriptFail`" 1>nul 2>nul"
    if ($LASTEXITCODE -eq 0) {
        throw "FAIL: expected controlled error when artifact is missing"
    }
    Write-Host "PASS: missing artifact correctly fails"
}
finally {
    if (Test-Path $obfDir) { Remove-Item $obfDir -Recurse -Force }
    Move-Item -Path $backupDir -Destination $obfDir
}

Write-Host "[5/5] Completed obfuscation acceptance checks"
