<#
.SYNOPSIS
  Slim the MePhC-Windows working tree: delete bulky generated audit data, keep code.

.DESCRIPTION
  Dry run by default; pass -Apply to delete. Everything removed here is
  tracked in git and already pushed to GitHub (WD-nanophotonics/MePhC,
  branch sandbox), so it remains recoverable with
  `git checkout -- <path>` or a fresh clone. The deletions are left
  uncommitted; do not commit them unless you also want them gone upstream.
  Findings from these files are summarised in LegumePhC/docs/legacy/.

.PARAMETER Apply
  Actually delete. Without it the script only reports.

.EXAMPLE
  .\scripts\legacy_cleanup\mephc_windows_slim.ps1          # dry run
  .\scripts\legacy_cleanup\mephc_windows_slim.ps1 -Apply   # delete
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$Repo = 'C:\Users\icywo\PycharmProjects\MePhC-Windows'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath (Join-Path $Repo '.git'))) { throw "not a git checkout: $Repo" }

function Invoke-RepoGit { & git.exe -c safe.directory=* -C $Repo @args }

# --- Safety: clean, and local sandbox equals the pushed sandbox -----------
$status = @(Invoke-RepoGit status --porcelain --untracked-files=no)
$alreadySlim = $status | Where-Object { $_ -notmatch '^ D audit/' }
if ($alreadySlim) { throw "MePhC-Windows has local changes other than audit deletions:`n$($alreadySlim -join "`n")" }
$local = (Invoke-RepoGit rev-parse sandbox).Trim()
$remote = (Invoke-RepoGit rev-parse origin/sandbox).Trim()
if ($local -ne $remote) { throw "sandbox ($local) differs from origin/sandbox ($remote); push or reconcile first" }
Write-Host "MePhC-Windows sandbox == origin/sandbox @ $($local.Substring(0, 7))"
Write-Host ($(if ($Apply) { 'MODE: APPLY (deleting)' } else { 'MODE: dry run (nothing is deleted; pass -Apply to delete)' })) -ForegroundColor $(if ($Apply) { 'Yellow' } else { 'Cyan' })

$audit = Join-Path $Repo 'audit'
$targets = @()
$archive = Join-Path $audit 'infrastructure\local_replica_archive'
if (Test-Path -LiteralPath $archive) { $targets += Get-Item -LiteralPath $archive -Force }
foreach ($folder in 'e9d', 'e9e', 'e9f') {
    $dir = Join-Path $audit $folder
    if (Test-Path -LiteralPath $dir) {
        $targets += Get-ChildItem -LiteralPath $dir -Recurse -File -Include '*request_graph*.json', '*raw_result*.json', '*_validation.json' | Where-Object { $_.Length -ge 256KB }
    }
}
$targets += Get-ChildItem -LiteralPath (Join-Path $Repo 'mephc') -Recurse -Force -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue

$total = 0
foreach ($item in $targets) {
    $bytes = if ($item.PSIsContainer) { (Get-ChildItem -LiteralPath $item.FullName -Recurse -Force -File | Measure-Object Length -Sum).Sum } else { $item.Length }
    if ($null -eq $bytes) { $bytes = 0 }
    $total += $bytes
    Write-Host ("{0,9:N1} MB  {1}" -f ($bytes / 1MB), $item.FullName.Substring($Repo.Length + 1))
    if ($Apply) {
        if ($item.PSIsContainer) { Remove-Item -LiteralPath $item.FullName -Recurse -Force } else { Remove-Item -LiteralPath $item.FullName -Force }
    }
}
Write-Host ("Total: {0:N1} MB in {1} entries {2}" -f ($total / 1MB), $targets.Count, $(if ($Apply) { 'deleted' } else { 'would be deleted' }))
Write-Host 'Kept: all code (mephc/, tests/, tools/, vendor/), audit scripts and small result JSONs, and the git history.'
