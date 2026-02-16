param(
  [string]$Backend = $env:KAI_BACKEND_URL,
  [string]$AccountName = "QA MultiSheet",
  [string]$DataPrefix = "qa",
  [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Backend)) {
  $Backend = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io"
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $RunBase = Join-Path $RepoRoot "verification_runs"
  New-Item -ItemType Directory -Force -Path $RunBase | Out-Null
  $OutDir = Join-Path $RunBase ("multisheet_upload_" + $Stamp)
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Write-Json($Path, $Obj) {
  $Obj | ConvertTo-Json -Depth 10 | Set-Content -Path $Path -Encoding UTF8
}

$meta = [ordered]@{
  backend = $Backend
  account_name = $AccountName
  data_prefix = $DataPrefix
  started_at = (Get-Date).ToString("o")
  generated_file = $null
  upload = $null
  manifest = $null
  assertions = @()
  ok = $false
}

# 1) Generate a tiny deterministic XLSX with 7+ sheets.
$xlsxPath = Join-Path $OutDir "multisheet.xlsx"
$genScript = Join-Path $PSScriptRoot "gen_multisheet_xlsx.py"
if (-not (Test-Path $genScript)) {
  throw "Missing generator script: $genScript"
}
& python $genScript $xlsxPath | Out-Null
if (-not (Test-Path $xlsxPath)) {
  throw "Failed to generate XLSX fixture: $xlsxPath"
}
$meta.generated_file = $xlsxPath

# 2) Upload via multipart/form-data.
# Note: Windows PowerShell 5.1 lacks Invoke-WebRequest -Form, so we use curl.exe.
$uploadUrl = "$Backend/api/data/upload"
$uploadOut = & curl.exe -sS -X POST `
  -F ("account_name=" + $AccountName) `
  -F ("data_prefix=" + $DataPrefix) `
  -F ("files=@" + $xlsxPath) `
  $uploadUrl
if ([string]::IsNullOrWhiteSpace($uploadOut)) {
  throw "Upload returned empty response."
}
$uploadJson = $null
try { $uploadJson = $uploadOut | ConvertFrom-Json } catch { $uploadJson = @{ raw = $uploadOut } }
$meta.upload = $uploadJson

if (-not $uploadJson -or $uploadJson.status -ne "success") {
  throw "Upload failed (expected status=success)."
}

# 3) Inspect manifest (metadata-only) from backend.
$acctEsc = [uri]::EscapeDataString($AccountName)
$prefEsc = [uri]::EscapeDataString($DataPrefix)
$manifestUrl = "$Backend/api/data/manifest?account_name=$acctEsc&data_prefix=$prefEsc"
$manifestResp = Invoke-WebRequest -Method GET -Uri $manifestUrl -UseBasicParsing
if ($manifestResp.StatusCode -ne 200) {
  throw "Manifest endpoint failed: HTTP $($manifestResp.StatusCode)"
}
$manifestJson = $manifestResp.Content | ConvertFrom-Json
$meta.manifest = $manifestJson

function Assert($Name, $Ok, $Details) {
  $meta.assertions += [ordered]@{ name = $Name; ok = [bool]$Ok; details = $Details }
  if (-not $Ok) { throw "Assertion failed: $Name" }
}

Assert "manifest_status_success" ($manifestJson.status -eq "success") @{ status = $manifestJson.status }
Assert "manifest_has_entries" ($manifestJson.manifest.Count -ge 7) @{ count = $manifestJson.manifest.Count }

$sheets = @()
try {
  $sheets = $manifestJson.manifest | ForEach-Object { $_.sheet } | Where-Object { $_ -ne $null } | Select-Object -Unique
} catch {}
Assert "manifest_includes_campaigns_sheet" ($sheets -contains "Campaigns") @{ sheets = $sheets }
Assert "manifest_includes_adgroups_sheet" ($sheets -contains "AdGroups") @{ sheets = $sheets }
Assert "manifest_includes_ads_sheet" ($sheets -contains "Ads") @{ sheets = $sheets }
Assert "manifest_includes_keywords_sheet" ($sheets -contains "Keywords") @{ sheets = $sheets }
Assert "manifest_includes_search_terms_sheet" ($sheets -contains "SearchTerms") @{ sheets = $sheets }
Assert "manifest_includes_conversions_sheet" ($sheets -contains "Conversions") @{ sheets = $sheets }
Assert "manifest_includes_landing_pages_sheet" ($sheets -contains "LandingPages") @{ sheets = $sheets }

$meta.ok = $true
$meta.ended_at = (Get-Date).ToString("o")

Write-Json (Join-Path $OutDir "summary.json") $meta
Write-Output "Run folder: $OutDir"

