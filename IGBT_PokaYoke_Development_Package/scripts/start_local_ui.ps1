param(
    [int]$Port = 8765,
    [string]$DataDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) "data")
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$builtIndex = Join-Path $projectRoot "web\dist\index.html"
if (-not (Test-Path -LiteralPath $builtIndex -PathType Leaf)) {
    throw "React frontend build is missing. Run 'npm ci' and 'npm run build' from frontend/igbt-station-ui first."
}
$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:IGBT_UI_PORT = [string]$Port
$env:IGBT_DATA_DIR = [System.IO.Path]::GetFullPath($DataDirectory)
if (-not $env:IGBT_ADMIN_TOKEN) {
    $env:IGBT_ADMIN_TOKEN = Read-Host "Set a local administrator token"
}
if ([string]::IsNullOrWhiteSpace($env:IGBT_ADMIN_TOKEN)) { throw "IGBT_ADMIN_TOKEN must not be empty" }
if (-not $env:IGBT_STATION_TOKEN) {
    $env:IGBT_STATION_TOKEN = Read-Host "Set a separate station runtime token"
}
if ([string]::IsNullOrWhiteSpace($env:IGBT_STATION_TOKEN)) { throw "IGBT_STATION_TOKEN must not be empty" }
Write-Host "Starting shadow-only dashboard at http://127.0.0.1:$Port"
Write-Host "SQLite and evidence directory: $env:IGBT_DATA_DIR"
python -m igbt_pokayoke.ui_app
