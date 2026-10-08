param(
    [ValidateSet('Release', 'Debug')][string]$Configuration = 'Release',
    [switch]$SkipWebBuild
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$androidRoot = Join-Path $projectRoot 'apps\mobile-shell\android'
$versionSource = Get-Content -LiteralPath (Join-Path $androidRoot 'app\build.gradle') -Raw
$versionMatch = [regex]::Match($versionSource, '(?m)^\s*versionName\s+"([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?)"')
if (!$versionMatch.Success) { throw 'A semantic Android versionName is required before packaging.' }
$releaseVersion = $versionMatch.Groups[1].Value
$loadedLocalPassword = $false
function Checked([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Build command failed with exit code $LASTEXITCODE" }
}
Push-Location $projectRoot
try {
    if ($Configuration -eq 'Release' -and !$env:ADAM_KEYSTORE) {
        $privateDirectory = Join-Path $env:USERPROFILE '.android\adam-release'
        $credentialPath = Join-Path $privateDirectory 'password.xml'
        if (!(Test-Path -LiteralPath $credentialPath)) {
            throw 'Set ADAM_KEYSTORE, ADAM_STORE_PASSWORD and ADAM_KEY_ALIAS for your release signing identity.'
        }
        $securePassword = Import-Clixml -LiteralPath $credentialPath
        $env:ADAM_KEYSTORE = Join-Path $privateDirectory 'adam-release.jks'
        $env:ADAM_STORE_PASSWORD = [Net.NetworkCredential]::new('', $securePassword).Password
        $env:ADAM_KEY_ALIAS = 'adam-release'
        $loadedLocalPassword = $true
    }
    if (!$SkipWebBuild) { Checked { pnpm --filter '@adam/web' build } }
    # Repair Explorer's read-only flags in generated files, including directories.
    foreach ($relative in @('.gradle', 'build', 'app\build', 'app\src\main\assets', 'capacitor-cordova-android-plugins')) {
        $generatedPath = Join-Path $androidRoot $relative
        if (Test-Path -LiteralPath $generatedPath) {
            $items = @(Get-Item -LiteralPath $generatedPath -Force) + @(Get-ChildItem -LiteralPath $generatedPath -Recurse -Force)
            foreach ($item in $items) {
                if ($item.Attributes -band [IO.FileAttributes]::ReadOnly) {
                    $item.Attributes = $item.Attributes -band (-bnot [IO.FileAttributes]::ReadOnly)
                }
                # Explorer metadata is not an AAPT resource and can corrupt
                # previously compiled resource directories reused by Gradle.
                if (!$item.PSIsContainer -and $item.Name -ieq 'desktop.ini') {
                    Remove-Item -LiteralPath $item.FullName -Force
                }
            }
        }
    }
    Checked { pnpm --filter '@adam/mobile-shell' exec cap sync android }
    Push-Location $androidRoot
    try { Checked { .\gradlew.bat ":app:assemble$Configuration" --max-workers=2 --console=plain } }
    finally { Pop-Location }
    $variant = $Configuration.ToLowerInvariant()
    $apk = Join-Path $androidRoot "app\build\outputs\apk\$variant\app-$variant.apk"
    if (!(Test-Path -LiteralPath $apk)) { throw 'The expected signed APK was not produced.' }
    $releaseDirectory = Join-Path $projectRoot 'releases'
    New-Item -ItemType Directory -Force -Path $releaseDirectory | Out-Null
    $destination = Join-Path $releaseDirectory "ADAM-$releaseVersion-$variant.apk"
    Copy-Item -LiteralPath $apk -Destination $destination -Force
    $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $(Split-Path -Leaf $destination)" | Set-Content -LiteralPath "$destination.sha256" -Encoding ascii
    Write-Output "APK: $destination"
} finally {
    if ($loadedLocalPassword) {
        Remove-Item Env:ADAM_STORE_PASSWORD,Env:ADAM_KEYSTORE,Env:ADAM_KEY_ALIAS -ErrorAction SilentlyContinue
    }
    Pop-Location
}
