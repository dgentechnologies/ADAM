param()

# Apply ADAM branding to this project in Windows Explorer. Safe to run again
# after cloning or moving the project; the icon path is relative to the folder.
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$icon = Join-Path $project 'resources\icons\adam-folder.ico'
$metadata = Join-Path $project 'desktop.ini'
if (-not (Test-Path -LiteralPath $icon -PathType Leaf)) {
    throw "Missing ADAM folder icon: $icon"
}
$content = "[.ShellClassInfo]`r`nIconResource=resources\icons\adam-folder.ico,0`r`nInfoTip=ADAM Desktop - Windows companion project`r`n"
if (Test-Path -LiteralPath $metadata) {
    $file = Get-Item -LiteralPath $metadata -Force
    $file.Attributes = [System.IO.FileAttributes]::Normal
}
[System.IO.File]::WriteAllText($metadata, $content, [System.Text.Encoding]::Unicode)
(Get-Item -LiteralPath $metadata -Force).Attributes = [System.IO.FileAttributes]::Hidden -bor [System.IO.FileAttributes]::System
$directory = Get-Item -LiteralPath $project -Force
$directory.Attributes = $directory.Attributes -bor [System.IO.FileAttributes]::ReadOnly

if (-not ('AdamExplorerNotification' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class AdamExplorerNotification {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    public static extern void SHChangeNotify(uint eventId, uint flags, string item1, IntPtr item2);
}
'@
}
# Update just this folder's cached metadata, without restarting Explorer.
[AdamExplorerNotification]::SHChangeNotify(0x00002000, 0x0005, $project, [IntPtr]::Zero)
Write-Output 'ADAM Desktop folder icon applied.'
