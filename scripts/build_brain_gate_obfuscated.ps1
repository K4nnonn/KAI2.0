Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$modulePath = "services/api/kai_core/shared/brain_gate.py"
if (-not (Test-Path $modulePath)) {
    throw "brain_gate source not found: $modulePath"
}

Write-Host "Checking Nuitka..."
python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('nuitka') else 1)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Nuitka not found. Installing..."
    python -m pip install nuitka
    if ($LASTEXITCODE -ne 0) {
        throw "Nuitka install failed."
    }
}

$outDir = Join-Path $repoRoot "build/obf"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "Compiling brain_gate module..."
python -m nuitka --module $modulePath "--output-dir=$outDir" --zig --assume-yes-for-downloads
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka compilation failed."
}

$artifact = Get-ChildItem -Path $outDir -File | Where-Object {
    $_.Name -like "brain_gate*.pyd" -or $_.Name -like "brain_gate*.so"
} | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $artifact) {
    throw "Compiled artifact not found in $outDir"
}

$hash = (Get-FileHash -Path $artifact.FullName -Algorithm SHA256).Hash
$manifestPath = Join-Path $outDir "brain_gate_build_manifest.txt"
$pythonVersion = (python -c "import sys;print(sys.version.replace('\n',' '))")
$manifestLines = @(
    "built_at_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
    "python_version=$pythonVersion"
    "artifact_name=$($artifact.Name)"
    "artifact_path=$($artifact.FullName)"
    "artifact_sha256=$hash"
)
$manifestLines | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "Build completed."
Write-Host "Artifact: $($artifact.FullName)"
Write-Host "Manifest: $manifestPath"
