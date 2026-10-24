param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace ".venv\Scripts\python.exe"
$liveVariables = @{
    ENABLE_REAL_PAPER_SEARCH = "true"
    ENABLE_REAL_CVE_LOOKUP = "true"
    ENABLE_GITHUB_SEARCH = "true"
    ENABLE_REAL_RFC_LOOKUP = "true"
    ENABLE_SEMANTIC_SCHOLAR = "true"
    ENABLE_ARXIV = "true"
    ENABLE_NVD = "true"
}
$previous = @{}

foreach ($name in $liveVariables.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    [Environment]::SetEnvironmentVariable($name, $liveVariables[$name], "Process")
}

try {
    Set-Location $workspace
    & $python -m uvicorn backend.main:app --host $HostAddress --port $Port --workers 1
}
finally {
    foreach ($name in $liveVariables.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
    }
}
