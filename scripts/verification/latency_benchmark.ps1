param(
  [string]$Backend = "https://kai-api.happyrock-0c0fe8c1.eastus.azurecontainerapps.io",
  [string]$SessionId = "",
  [int]$Iterations = 3,
  [string]$OutPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SessionId)) {
  throw "SessionId is required for latency benchmark."
}

function Measure-Request($Method, $Url, $Body = $null) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  if ($Body -ne $null) {
    Invoke-RestMethod -Method $Method -Uri $Url -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 6) | Out-Null
  } else {
    Invoke-RestMethod -Method $Method -Uri $Url | Out-Null
  }
  $sw.Stop()
  return [math]::Round($sw.Elapsed.TotalSeconds, 3)
}

$routeUrl = "$Backend/api/chat/route"
$sendUrl = "$Backend/api/chat/send"

$routeTimes = @()
$sendTimes = @()

for ($i = 0; $i -lt $Iterations; $i++) {
  $routeTimes += Measure-Request "POST" $routeUrl @{ message = "What can you do?"; session_id = $SessionId }
  $sendTimes += Measure-Request "POST" $sendUrl @{ message = "What can you do?"; ai_enabled = $true; session_id = $SessionId }
}

function Stats($arr) {
  $sorted = $arr | Sort-Object
  $avg = [math]::Round(($arr | Measure-Object -Average).Average, 3)
  $p95Index = [math]::Ceiling($sorted.Count * 0.95) - 1
  if ($p95Index -lt 0) { $p95Index = 0 }
  $p95 = $sorted[$p95Index]
  return [ordered]@{
    avg = $avg
    min = $sorted[0]
    max = $sorted[-1]
    p95 = $p95
    samples = $sorted
  }
}

$result = [ordered]@{
  timestamp = (Get-Date).ToString("o")
  backend = $Backend
  iterations = $Iterations
  route = (Stats $routeTimes)
  send = (Stats $sendTimes)
}

if ($OutPath) {
  $result | ConvertTo-Json -Depth 6 | Set-Content -Path $OutPath -Encoding UTF8
}

$result | ConvertTo-Json -Depth 6
