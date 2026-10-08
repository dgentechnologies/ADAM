import { chmod, readdir } from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// Windows Explorer may mark generated directories/desktop.ini read-only.
// Next 15's cleanup can retry forever on those paths. Only fix generated output.
async function writable(directory) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === 'ENOENT') return;
    throw error;
  }
  await chmod(directory, 0o755);
  for (const entry of entries) {
    if (entry.isSymbolicLink()) continue;
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) await writable(target);
    else await chmod(target, 0o644);
  }
}
if (process.platform === 'win32') {
  // chmod does not reliably clear Windows directory attributes. Use native
  // attributes before Next recursively removes its generated output.
  execFileSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      `
    $ErrorActionPreference = 'Stop'
    foreach ($name in @('.next', 'out')) {
      $target = [IO.Path]::GetFullPath((Join-Path $env:ADAM_WEB_BUILD_ROOT $name))
      if (!$target.StartsWith($env:ADAM_WEB_BUILD_ROOT + [IO.Path]::DirectorySeparatorChar)) { throw 'Unexpected generated path' }
      if (Test-Path -LiteralPath $target) {
        $items = @(Get-Item -LiteralPath $target -Force) + @(Get-ChildItem -LiteralPath $target -Recurse -Force)
        foreach ($item in $items) {
          if ($item.Attributes -band [IO.FileAttributes]::ReadOnly) {
            $item.Attributes = $item.Attributes -band (-bnot [IO.FileAttributes]::ReadOnly)
          }
        }
        if ($name -eq 'out') { Remove-Item -LiteralPath $target -Recurse -Force }
      }
    }
  `,
    ],
    { windowsHide: true, stdio: 'inherit', env: { ...process.env, ADAM_WEB_BUILD_ROOT: webRoot } },
  );
  await writable(path.join(webRoot, '.next'));
}
