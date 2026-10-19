param(
    [ValidateSet('evomolsteer_deepseek', 'replay')]
    [string]$Backend = 'evomolsteer_deepseek',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path $PSScriptRoot -Parent
if (-not $Output) {
    $Output = Join-Path $taskRepo ('outputs/ck2_' + $Backend + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if ($Backend -eq 'evomolsteer_deepseek' -and -not $env:DEEPSEEK_API_KEY) {
    $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')
}
Push-Location $taskRepo
try {
    & uv run --frozen --extra syncraft --extra evomolsteer agenticgen run --config examples/ck2_top3/config.json --backend $Backend --output $Output
    if ($LASTEXITCODE -ne 0) { throw "AgenticGen exited with status $LASTEXITCODE. Inspect output receipts." }
} finally {
    Pop-Location
}
