param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArguments
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv and install requirements-dev.txt as described in README.md.'
}
& $python (Join-Path $project 'src\app.py') @AppArguments
exit $LASTEXITCODE
