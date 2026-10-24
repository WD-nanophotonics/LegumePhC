<#
.SYNOPSIS
  Remove generated MePhC runner data and caches from the WSL home (code is kept).

.DESCRIPTION
  Dry run by default: prints every target and whether it exists. Nothing is
  deleted unless -Apply is given. Works from Windows through \\wsl.localhost;
  it never runs wsl.exe. Close WSL shells/VS Code remotes using these folders
  first, and make sure the MePhC runner is not running.

  Before deleting the 433 runner worktrees, every worktree HEAD commit is
  pinned as refs/archive/worktrees/<sha> in the kept bare repository
  ~/.cache/mephc-runner/MEPHC.git, so no commit becomes unreachable.

  Kept: science results/datasets/input bundles, ~/.local/share/mephc-archive,
  the MEPHC.git bare repo, the conda env 'mp', ~/.vscode-server, and every
  project folder (TriLatt, SqrLatt, GratingCoupler, CBS_readonly,
  CoupledComponents, AIInbox). Legacy findings are summarised in
  LegumePhC/docs/legacy/.

.PARAMETER Apply
  Actually delete. Without it the script only reports.

.PARAMETER Measure
  Also measure sizes (slow over \\wsl.localhost; can take tens of minutes).

.EXAMPLE
  .\scripts\legacy_cleanup\wsl_cleanup.ps1             # dry run
  .\scripts\legacy_cleanup\wsl_cleanup.ps1 -Measure    # dry run with sizes
  .\scripts\legacy_cleanup\wsl_cleanup.ps1 -Apply      # delete
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$Measure,
    [string]$WslHome = '\\wsl.localhost\Ubuntu\home\icy'
)

$ErrorActionPreference = 'Stop'

function Join([string]$relative) { Join-Path $WslHome $relative }

function Size-Of([string]$path) {
    if (-not $Measure -or -not (Test-Path -LiteralPath $path)) { return $null }
    $sum = (Get-ChildItem -LiteralPath $path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { $sum = 0 }
    return [math]::Round($sum / 1GB, 2)
}

$script:Failures = New-Object System.Collections.Generic.List[string]

function Remove-Link([System.IO.FileSystemInfo]$item) {
    # Linux symlinks appear over \\wsl.localhost as reparse points whose
    # targets Windows often cannot resolve; delete the link itself.
    try {
        if ($item -is [System.IO.DirectoryInfo]) { [System.IO.Directory]::Delete($item.FullName, $false) } else { [System.IO.File]::Delete($item.FullName) }
    } catch {
        $quoted = '"' + $item.FullName + '"'
        cmd.exe /s /c "del /f /q $quoted 2>nul & rmdir /q $quoted 2>nul" | Out-Null
        if ([System.IO.File]::Exists($item.FullName) -or [System.IO.Directory]::Exists($item.FullName)) { throw }
    }
}

function Remove-Entry([System.IO.FileSystemInfo]$item) {
    try {
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            Remove-Link $item
        } elseif ($item -is [System.IO.DirectoryInfo]) {
            foreach ($child in $item.GetFileSystemInfos()) { Remove-Entry $child }
            [System.IO.Directory]::Delete($item.FullName, $false)
        } else {
            if ($item.Attributes -band [System.IO.FileAttributes]::ReadOnly) { $item.Attributes = [System.IO.FileAttributes]::Normal }
            [System.IO.File]::Delete($item.FullName)
        }
    } catch {
        $script:Failures.Add("$($item.FullName): $($_.Exception.Message)")
    }
}

function Remove-Tree([string]$path) {
    # Entry by entry: never follows links, and one stubborn entry is reported
    # instead of aborting the whole cleanup.
    $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    if ($null -ne $item) { Remove-Entry $item }
}

if (-not (Test-Path -LiteralPath $WslHome)) { throw "WSL home not reachable: $WslHome (start the Ubuntu distro once)" }

$bare = Join '.cache\mephc-runner\MEPHC.git'
$checkouts = Join '.cache\mephc-runner\checkouts'

# Targets: path, approximate size from the 2026-10-24 survey, description.
$targets = @(
    @{ Path = $checkouts; Approx = '29 GB'; What = 'MePhC runner detached worktrees (433 checkouts)' },
    @{ Path = (Join '.local\state\mephc-runner\MEPHC\runner\jobs'); Approx = '7.1 GB'; What = 'runner job scratch' },
    @{ Path = (Join '.local\state\mephc-runner\MEPHC\flow\datasets'); Approx = '1.6 GB'; What = 'runner flow datasets' },
    @{ Path = (Join 'miniconda3\pkgs'); Approx = '2.3 GB'; What = 'conda package cache (contents only; envs keep their hard links)'; ContentsOnly = $true },
    @{ Path = (Join 'Miniconda3-latest-Linux-aarch64.sh'); Approx = '164 MB'; What = 'leftover installer' },
    @{ Path = (Join 'miniconda.sh'); Approx = '163 MB'; What = 'leftover installer' },
    @{ Path = (Join '.codex\.tmp'); Approx = '65 MB'; What = 'Codex temp files' }
)

# Small caches inside project folders (code is untouched).
$projectDirs = 'TriLatt', 'SqrLatt', 'GratingCoupler', 'CBS_readonly', 'CoupledComponents', 'AIInbox' | ForEach-Object { Join $_ } | Where-Object { Test-Path -LiteralPath $_ }
$caches = foreach ($dir in $projectDirs) {
    Get-ChildItem -LiteralPath $dir -Recurse -Force -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -in '__pycache__', '.pytest_cache' } | ForEach-Object { $_.FullName }
    Get-ChildItem -LiteralPath $dir -Recurse -Force -File -Filter '*Zone.Identifier' -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
}

Write-Host "WSL home: $WslHome"
Write-Host ($(if ($Apply) { 'MODE: APPLY (deleting)' } else { 'MODE: dry run (nothing is deleted; pass -Apply to delete)' })) -ForegroundColor $(if ($Apply) { 'Yellow' } else { 'Cyan' })
Write-Host ''

# --- Safety: the runner must be idle --------------------------------------
$jobs = Join '.local\state\mephc-runner\MEPHC\runner\jobs'
if (Test-Path -LiteralPath $jobs) {
    $latest = Get-ChildItem -LiteralPath $jobs -Force -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest -and $latest.LastWriteTime -gt (Get-Date).AddHours(-24)) {
        throw "runner jobs changed within 24 h ($($latest.LastWriteTime)); is the MePhC runner still active? Stop it first."
    }
}

# --- Safety: pin every worktree HEAD in the kept bare repo ------------------
$heads = @()
if (Test-Path -LiteralPath (Join-Path $bare 'worktrees')) {
    $heads = Get-ChildItem -LiteralPath (Join-Path $bare 'worktrees') -Directory | ForEach-Object {
        (Get-Content -LiteralPath (Join-Path $_.FullName 'HEAD') -TotalCount 1).Trim()
    } | Where-Object { $_ -match '^[0-9a-f]{40}$' } | Sort-Object -Unique
}
Write-Host "Worktree HEAD commits to pin in MEPHC.git: $($heads.Count)"
if ($heads.Count -and (Test-Path -LiteralPath $checkouts)) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git is required to pin worktree commits' }
    $missing = @($heads | Where-Object { (git -c safe.directory=* -C $bare cat-file -t $_ 2>$null) -ne 'commit' })
    if ($missing.Count) { throw "bare repo is missing $($missing.Count) worktree commits (e.g. $($missing[0])); not deleting checkouts" }
    if ($Apply) {
        # Feed git a BOM-free, LF-terminated file through cmd redirection:
        # Windows PowerShell's pipeline and process stdin add CRLF and a BOM,
        # which `git update-ref --stdin` rejects.
        $text = (($heads | ForEach-Object { "update refs/archive/worktrees/$_ $_" }) -join "`n") + "`n"
        $commandFile = [System.IO.Path]::GetTempFileName()
        try {
            [System.IO.File]::WriteAllBytes($commandFile, [System.Text.Encoding]::ASCII.GetBytes($text))
            $gitOutput = cmd.exe /s /c "git -c safe.directory=* -C `"$bare`" update-ref --stdin < `"$commandFile`" 2>&1"
            if ($LASTEXITCODE -ne 0) { throw "pinning worktree commits failed: $gitOutput" }
        } finally {
            Remove-Item -LiteralPath $commandFile -Force -ErrorAction SilentlyContinue
        }
        $pinned = @(git -c safe.directory=* -C $bare for-each-ref --format='%(refname)' refs/archive/worktrees).Count
        if ($pinned -lt $heads.Count) { throw "only $pinned of $($heads.Count) commits pinned" }
        Write-Host "  pinned $pinned commits under refs/archive/worktrees/" -ForegroundColor Green
    } else {
        Write-Host '  (would pin each as refs/archive/worktrees/<sha>)'
    }
}
Write-Host ''

# --- Report / delete ------------------------------------------------------
foreach ($target in $targets) {
    $exists = Test-Path -LiteralPath $target.Path
    $size = Size-Of $target.Path
    $label = if ($null -ne $size) { "$size GB" } else { "~$($target.Approx)" }
    $state = if ($exists) { 'present' } else { 'absent ' }
    Write-Host ("[{0}] {1,-9} {2}" -f $state, $label, $target.Path)
    Write-Host ("            {0}" -f $target.What) -ForegroundColor DarkGray
    if ($Apply -and $exists) {
        if ($target.ContentsOnly) {
            foreach ($child in (Get-Item -LiteralPath $target.Path -Force).GetFileSystemInfos()) { Remove-Entry $child }
        } else {
            Remove-Tree $target.Path
        }
        $left = Test-Path -LiteralPath $target.Path
        if ($target.ContentsOnly) { $left = @((Get-Item -LiteralPath $target.Path -Force).GetFileSystemInfos()).Count -gt 0 }
        if ($left) { Write-Host '            partly deleted (see summary)' -ForegroundColor Yellow } else { Write-Host '            deleted' -ForegroundColor Green }
    }
}
Write-Host ("[caches ] {0} __pycache__/.pytest_cache/Zone.Identifier entries in project folders" -f @($caches).Count)
if ($Apply) { foreach ($cache in $caches) { Remove-Tree $cache } }

if ($Apply -and (Test-Path -LiteralPath $bare)) {
    git -c safe.directory=* -C $bare worktree prune 2>$null
    Write-Host 'Pruned stale worktree metadata in MEPHC.git (pinned refs keep every commit).' -ForegroundColor Green
}

if ($script:Failures.Count) {
    Write-Host ''
    Write-Host "Could not delete $($script:Failures.Count) entries (first 10 below). Rerunning is safe; anything left" -ForegroundColor Yellow
    Write-Host 'can also be removed from your Ubuntu terminal, e.g.  rm -rf ~/miniconda3/pkgs/*' -ForegroundColor Yellow
    $script:Failures | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkYellow }
}

Write-Host ''
Write-Host 'Kept: science results/datasets, mephc-archive, MEPHC.git, conda env mp, .vscode-server, all project folders.'
Write-Host 'To reclaim the space on the Windows drive afterwards, compact the WSL virtual disk (ext4.vhdx)'
Write-Host 'from an elevated prompt after `wsl --shutdown`, e.g. Optimize-VHD or `diskpart` compact vdisk.'
