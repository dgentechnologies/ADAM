#requires -Version 7.4
param(
    [Parameter(Mandatory = $true)]
    [string]$PreviousRoot
)

# pnpm uses absolute Windows junctions. After a workspace is moved, retarget
# only its internal links; never remove packages or follow junction targets.
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$oldRoot = [System.IO.Path]::GetFullPath($PreviousRoot).TrimEnd('\')
$oldPrefix = $oldRoot + '\'
$newPrefix = $project + '\'
$pending = [System.Collections.Generic.Stack[string]]::new()
$moduleRoots = @((Join-Path $project 'node_modules'))
foreach ($group in @('apps', 'packages')) {
    $groupPath = Join-Path $project $group
    if (Test-Path -LiteralPath $groupPath) {
        foreach ($workspace in Get-ChildItem -LiteralPath $groupPath -Directory) {
            $moduleRoots += Join-Path $workspace.FullName 'node_modules'
        }
    }
}
$virtualStore = Join-Path $project 'node_modules\.pnpm'
if (Test-Path -LiteralPath $virtualStore) {
    $moduleRoots += Join-Path $virtualStore 'node_modules'
    foreach ($package in Get-ChildItem -LiteralPath $virtualStore -Directory) {
        if (-not ($package.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
            $moduleRoots += Join-Path $package.FullName 'node_modules'
        }
    }
}
foreach ($moduleRoot in $moduleRoots | Select-Object -Unique) {
    if (Test-Path -LiteralPath $moduleRoot -PathType Container) { $pending.Push($moduleRoot) }
}
$repaired = 0
while ($pending.Count -gt 0) {
    $directory = $pending.Pop()
    foreach ($item in Get-ChildItem -LiteralPath $directory -Directory -Force) {
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            if ($item.LinkType -ne 'Junction') { continue }
            $target = [string]$item.Target
            if (-not $target.StartsWith($oldPrefix, [StringComparison]::OrdinalIgnoreCase)) { continue }
            $replacement = [System.IO.Path]::GetFullPath((Join-Path $project $target.Substring($oldPrefix.Length)))
            if (-not $item.FullName.StartsWith($newPrefix, [StringComparison]::OrdinalIgnoreCase) -or
                -not $replacement.StartsWith($newPrefix, [StringComparison]::OrdinalIgnoreCase) -or
                -not (Test-Path -LiteralPath $replacement -PathType Container)) {
                throw "Cannot safely retarget dependency: $($item.FullName)"
            }
            New-Item -ItemType Junction -Path $item.FullName -Target $replacement -Force | Out-Null
            $repaired++
        } elseif ($item.Name.StartsWith('@')) {
            # Scoped package containers can contain links. Real package source,
            # generated builds and local backups do not need traversal.
            $pending.Push($item.FullName)
        }
    }
}
$modulesFile = Join-Path $project 'node_modules\.modules.yaml'
if (Test-Path -LiteralPath $modulesFile) {
    $original = [System.IO.File]::ReadAllText($modulesFile)
    $updated = $original.Replace($oldRoot, $project).Replace($oldRoot.Replace('\', '/'), $project.Replace('\', '/'))
    if ($updated -ne $original) {
        [System.IO.File]::WriteAllText($modulesFile, $updated, [System.Text.UTF8Encoding]::new($false))
    }
}
Write-Output "Repaired $repaired internal dependency junctions. Package contents and versions preserved."
