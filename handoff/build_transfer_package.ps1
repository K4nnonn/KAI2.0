$ErrorActionPreference = "Stop"

$Source = "C:\Software Builds\repo_shadow\repo"
$OutRoot = "C:\Software Builds\handoff_packages"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Dest = Join-Path $OutRoot "transfer_package_$Stamp"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$ExcludeDirs = @(
  ".git", "node_modules", "dist", "build", ".next", "coverage",
  ".venv", "venv", "__pycache__", ".pytest_cache",
  ".idea", ".vscode",
  "integrity_runs", "logs", "tmp", "temp", "playwright-run",
  "tests\\e2e",
  "diagnostics"
)
$ExcludeFiles = @(
  ".env", ".env.*", "*.local", "*.log",
  "*.pem", "*.pfx", "*.key", "*.crt",
  "*secrets*", "*token*", "*credential*", "*apikey*", "*api_key*"
)

$XD = @()
foreach ($d in $ExcludeDirs) { $XD += @("/XD", (Join-Path $Source $d)) }
$XF = @()
foreach ($f in $ExcludeFiles) { $XF += @("/XF", $f) }

$rc = robocopy $Source $Dest /MIR /R:1 /W:1 /NFL /NDL /NP /NJH /NJS /NC /NS @XD @XF
if ($LASTEXITCODE -gt 7) { throw "robocopy failed: $LASTEXITCODE" }

Get-ChildItem -Path $Dest -Directory -Recurse -Force |
  Where-Object { $ExcludeDirs -contains $_.Name } |
  ForEach-Object { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }

foreach ($pattern in $ExcludeFiles) {
  Get-ChildItem -Path $Dest -File -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like $pattern } |
    ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
}

# Explicitly remove e2e artifacts from transfer package
$e2eDest = Join-Path $Dest "tests\\e2e"
if (Test-Path $e2eDest) {
  Remove-Item $e2eDest -Recurse -Force -ErrorAction SilentlyContinue
}

$handoff = "C:\Software Builds\repo_shadow\repo\diagnostics\handoff"
if (Test-Path $handoff) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Dest "handoff") | Out-Null
  Copy-Item -Path (Join-Path $handoff "*") -Destination (Join-Path $Dest "handoff") -Recurse -Force
}

$EnvExample = Join-Path $Dest ".env.example"
@"
# Example environment variables (DO NOT COMMIT REAL SECRETS)
KAI_BACKEND_URL=https://<your-backend-host>
KAI_FRONTEND_URL=https://<your-frontend-host>
KAI_ACCESS_PASSWORD=<set-me>
KAI_ENV_GUI_PASSWORD=<set-me>

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=

# Local LLM (Ollama)
LOCAL_LLM_ENDPOINT=http://localhost:11434
LOCAL_LLM_MODEL=qwen2.5:14b

# Storage / queues
AZURE_STORAGE_CONNECTION_STRING=
AZURE_STORAGE_ACCOUNT=
AZURE_STORAGE_CONTAINER=

# SA360 (OAuth-only)
SA360_CLIENT_ID=
SA360_CLIENT_SECRET=
SA360_REFRESH_TOKEN=
SA360_LOGIN_CUSTOMER_ID=
SA360_OAUTH_REDIRECT_URI=

# Feature flags
ADS_FETCH_ENABLED=false
"@ | Set-Content -Path $EnvExample -Encoding UTF8

$SecretReport = Join-Path $Dest "SECRET_SCAN_REPORT.txt"
"Secret scan report for $Dest" | Set-Content $SecretReport -Encoding UTF8
$Suspicious = @(
  "BEGIN PRIVATE KEY", "AKIA", "AIza", "xoxb-", "xoxp-", "ghp_",
  "password=", "secret=", "token=", "api_key", "client_secret"
)
 $secretHits = New-Object System.Collections.Generic.List[string]
foreach ($s in $Suspicious) {
  Get-ChildItem -Path $Dest -File -Recurse -ErrorAction SilentlyContinue |
    Select-String -Pattern $s -ErrorAction SilentlyContinue |
    ForEach-Object { $secretHits.Add(("{0}:{1} :: {2}" -f $_.Path, $_.LineNumber, $_.Line)) }
}
if ($secretHits.Count -gt 0) {
  $secretHits | Add-Content $SecretReport
}

$Manifest = Join-Path $Dest "MANIFEST_SHA256.txt"
$fileList = Get-ChildItem -Path $Dest -File -Recurse -Force |
  Where-Object { $_.FullName -ne $Manifest } |
  Sort-Object FullName
@("SHA256 manifest for $Dest") + ($fileList | ForEach-Object {
  $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
  "{0}  {1}" -f $hash, ($_.FullName.Substring($Dest.Length + 1))
}) | Set-Content $Manifest -Encoding UTF8

Write-Host "Created transfer package:" $Dest
Write-Host "Secret report:" $SecretReport
Write-Host "Manifest:" $Manifest
