$ErrorActionPreference = "Stop"

if (-not $env:MLFLOW_DB_USER) {
    $env:MLFLOW_DB_USER = Read-Host "PostgreSQL user untuk MLflow (contoh: mlflow_user)"
}
$securePassword = Read-Host "Password PostgreSQL untuk user $($env:MLFLOW_DB_USER)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    $encodedPassword = [System.Uri]::EscapeDataString($plainPassword)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

$dbHost = if ($env:MLFLOW_DB_HOST) { $env:MLFLOW_DB_HOST } else { "127.0.0.1" }
$dbPort = if ($env:MLFLOW_DB_PORT) { $env:MLFLOW_DB_PORT } else { "5432" }
$dbName = if ($env:MLFLOW_DB_NAME) { $env:MLFLOW_DB_NAME } else { "mlflow_tracking" }
$userEncoded = [System.Uri]::EscapeDataString($env:MLFLOW_DB_USER)
$backendUri = "postgresql+psycopg2://${userEncoded}:${encodedPassword}@${dbHost}:${dbPort}/${dbName}"

$serverData = Join-Path $PSScriptRoot "server-data"
New-Item -ItemType Directory -Force -Path $serverData | Out-Null
Push-Location $serverData
try {
    Write-Host "MLflow UI: http://127.0.0.1:5000"
    Write-Host "Metadata DB: ${dbHost}:${dbPort}/${dbName}"
    Write-Host "Artifact directory: $serverData\\mlartifacts"
    Write-Host "Keep this terminal open. Press Ctrl+C to stop."
    mlflow server --backend-store-uri $backendUri --host 127.0.0.1 --port 5000
} finally {
    Pop-Location
    $plainPassword = $null
    $encodedPassword = $null
    $backendUri = $null
}
