Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

Write-Host "Repository root: $repoRoot"

$uvExists = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)
if (-not $uvExists) {
    Write-Host "uv not found. Installing uv with Astral installer..."
    powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$HOME\.local\bin;$env:Path"
}

$uvExistsAfter = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)
if (-not $uvExistsAfter) {
    Write-Error "uv is still not available on PATH after install. Open a new shell and run this installer again."
}

if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists at .venv; reusing it."
}
else {
    Write-Host "Creating virtual environment at .venv ..."
    uv venv .venv
}

Write-Host "Installing requirements into .venv ..."
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt

Write-Host ""
Write-Host "Install complete."
Write-Host "Next steps:"
Write-Host "  1) Activate venv: .\.venv\Scripts\Activate.ps1"
Write-Host "  2) Run script:    python .\script\gd_1d_torch.py"
Write-Host "  3) Run tests:     pytest"
