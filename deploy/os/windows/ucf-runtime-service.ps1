# Template wrapper for Windows service registration (optional).
# No downloader logic included; adapt to your local service manager (sc.exe or NSSM).

param(
  [string]$BundleRoot = "$PSScriptRoot\..\..\bundles\current"
)

$env:UCF_BUNDLE_ROOT = $BundleRoot
$binary = Join-Path $BundleRoot 'bin\ucf-runtime.exe'

Write-Output "Use your service manager to run: $binary"
Write-Output "WorkingDirectory: $BundleRoot"
