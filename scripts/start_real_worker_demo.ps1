param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "RiskItem real Worker demo startup"
Write-Host "Root: $Root"

Set-Location $Root

$env:WORKFLOW_USE_CELERY = "true"
$env:WORKFLOW_DEMO_MODE = "true"
$env:WORKFLOW_CHECKPOINTER = "memory"
$env:MODELOPS_API_BASE = "http://127.0.0.1:$ApiPort"

Write-Host "Starting API on port $ApiPort"
Start-Process -FilePath "python" `
    -ArgumentList "-m uvicorn apps.modelops_api.main:app --host 0.0.0.0 --port $ApiPort" `
    -WorkingDirectory $Root `
    -WindowStyle Hidden

Start-Sleep -Seconds 4

Write-Host "Starting Celery worker"
Start-Process -FilePath "celery" `
    -ArgumentList "-A workers.app worker --loglevel=info --pool=solo" `
    -WorkingDirectory $Root `
    -WindowStyle Hidden

Write-Host "Starting Next.js frontend on port $WebPort"
Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run dev -- --port $WebPort" `
    -WorkingDirectory (Join-Path $Root "apps\web") `
    -WindowStyle Hidden

Write-Host ""
Write-Host "API:      http://127.0.0.1:$ApiPort"
Write-Host "Frontend: http://127.0.0.1:$WebPort"
Write-Host ""
Write-Host "Run this after the services are ready:"
Write-Host "python tests\test_e2e_celery_real.py"
