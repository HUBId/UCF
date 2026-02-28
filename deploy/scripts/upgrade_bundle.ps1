param(
  [Parameter(Mandatory = $true)][ValidateSet('upgrade','rollback')] [string]$Action,
  [Parameter(Mandatory = $true)] [string]$BundleId,
  [string]$HealthCommand = '.\\bin\\ucf-ops health check --bundle . --out .\\out\\health.json'
)

$Root = Get-Location
$BundlesDir = Join-Path $Root 'bundles'
$CurrentLink = Join-Path $BundlesDir 'current'
$PreviousLink = Join-Path $BundlesDir 'previous'
$Target = Join-Path $BundlesDir "releases/$BundleId"

New-Item -ItemType Directory -Path (Join-Path $BundlesDir 'releases') -Force | Out-Null
if (-not (Test-Path $Target)) {
  throw "missing target bundle: $Target"
}

if (Test-Path $CurrentLink) {
  $currentReal = (Get-Item $CurrentLink).Target
  if (Test-Path $PreviousLink) { Remove-Item $PreviousLink -Force }
  New-Item -ItemType Junction -Path $PreviousLink -Target $currentReal | Out-Null
}

switch ($Action) {
  'upgrade' {
    if (Test-Path $CurrentLink) { Remove-Item $CurrentLink -Force }
    New-Item -ItemType Junction -Path $CurrentLink -Target $Target | Out-Null
  }
  'rollback' {
    if (-not (Test-Path $PreviousLink)) { throw 'no previous bundle' }
    $prevReal = (Get-Item $PreviousLink).Target
    if (Test-Path $CurrentLink) { Remove-Item $CurrentLink -Force }
    New-Item -ItemType Junction -Path $CurrentLink -Target $prevReal | Out-Null
  }
}

Set-Location $CurrentLink
Invoke-Expression $HealthCommand
Write-Output "status=ok action=$Action current=$((Get-Location).Path)"
