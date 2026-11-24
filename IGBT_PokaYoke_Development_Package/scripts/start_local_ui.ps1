param(
    [int]$Port = 8765,
    [string]$WorkMode = "debug",
    [string]$DataDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) "data")
)
$ErrorActionPreference = "Stop"
$pkgRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $pkgRoot
$builtIndex = Join-Path $pkgRoot "web\dist\index.html"
if (-not (Test-Path -LiteralPath $builtIndex -PathType Leaf)) {
    throw "React frontend build is missing. Run 'npm ci' and 'npm run build' from frontend/igbt-station-ui first."
}
$serverExe = Join-Path $repoRoot "cpp_backend\build\igbt_station_server.exe"
if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
    throw "C++20 server binary is missing at $serverExe. Build cpp_backend via CMake first."
}
$env:PATH = "D:\msys2\ucrt64\bin;$env:PATH"
$env:IGBT_UI_PORT = [string]$Port
$env:IGBT_DATA_DIR = [System.IO.Path]::GetFullPath($DataDirectory)
$env:IGBT_WEB_DIST_DIR = Join-Path $pkgRoot "web\dist"
$env:IGBT_ENABLE_LIVE_CAMERA = "1"
$env:IGBT_WORK_MODE = $WorkMode
if (-not $env:IGBT_ADMIN_TOKEN) {
    $env:IGBT_ADMIN_TOKEN = Read-Host "Set a local administrator token"
}
if ([string]::IsNullOrWhiteSpace($env:IGBT_ADMIN_TOKEN)) { throw "IGBT_ADMIN_TOKEN must not be empty" }
if (-not $env:IGBT_STATION_TOKEN) {
    $env:IGBT_STATION_TOKEN = Read-Host "Set a separate station runtime token"
}
if ([string]::IsNullOrWhiteSpace($env:IGBT_STATION_TOKEN)) { throw "IGBT_STATION_TOKEN must not be empty" }
Write-Host "Starting C++20 native vision workstation (1280x720 @ 25 FPS) at http://127.0.0.1:$Port"
Write-Host "SQLite and evidence directory: $env:IGBT_DATA_DIR"
& $serverExe
