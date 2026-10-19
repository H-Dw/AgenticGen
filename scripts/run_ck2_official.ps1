param(
    [string]$Config = 'examples/ck2_official/dual.json',
    [string]$Output = 'outputs/ck2_official_release_example'
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path $PSScriptRoot -Parent
if (-not $env:DEEPSEEK_API_KEY) {
    $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')
}
Push-Location $taskRepo
try {
    & .venv/Scripts/python.exe -m agenticgen run --config $Config --output $Output
    if ($LASTEXITCODE -ne 0) { throw "AgenticGen exited with status $LASTEXITCODE. Inspect saved artifacts." }
} finally {
    Pop-Location
}
