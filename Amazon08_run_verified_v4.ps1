[CmdletBinding()]
param(
    [ValidateSet('ALL','SETUP','TESTS','SYNTHETIC','M5','PACKAGE')]
    [string]$Mode = 'ALL',

    [string]$RepoRoot = $PSScriptRoot,

    [string]$OutputRoot = '',

    [string]$M5Input = $env:AMAZON08_M5_INPUT
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'Continue'

# Amazon08 — State-Aligned Stockout Inventory System
# Windows local controller, CPU core path.
# Core does not require PyTorch/CUDA. GPU/Chronos is intentionally excluded.

$ProjectName = 'Amazon08_state_aligned_stockout_inventory_system'

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    throw '[P00_PREFLIGHT] RepoRoot cannot be empty.'
}

if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
    throw "[P00_PREFLIGHT] RepoRoot does not exist or is not a directory: $RepoRoot"
}

$Repo = (Resolve-Path -LiteralPath $RepoRoot).Path

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path (
        Split-Path -Parent $Repo
    ) ($ProjectName + '_outputs')
}

$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

$RunId = 'local-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$RunRoot = Join-Path $OutputRoot (Join-Path '.local-run' $RunId)
$LogRoot = Join-Path $RunRoot 'logs'
$ReportRoot = Join-Path $RunRoot 'reports'
$ConfigRoot = Join-Path $RunRoot 'configs'
$ProbeRoot = Join-Path $RunRoot 'probes'
$PatchRoot = Join-Path $RunRoot 'patches'
$StatePath = Join-Path $RunRoot 'state.json'

New-Item -ItemType Directory -Force -Path @($RunRoot,$LogRoot,$ReportRoot,$ConfigRoot,$ProbeRoot,$PatchRoot) | Out-Null

$env:PYTHONUTF8 = '1'
$env:PYTHONHASHSEED = '0'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:INVENTORY_AI_PROGRESS = '1'

$Stages = [ordered]@{}
$CurrentStage = 'P00_PREFLIGHT'
$PythonLauncher = $null
$PythonPrefix = @()
$Venv = $null
$VenvPython = $null
$M5Zip = $null
$LocalM5Config = $null
$PackageZip = $null

function Write-Stage {
    param([string]$Stage,[string]$Message)
    $line = '[{0}] [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$Stage,$Message
    Write-Host $line
    Add-Content -LiteralPath (Join-Path $LogRoot 'controller.log') -Value $line -Encoding utf8
}

function Save-State {
    param([string]$Stage,[ValidateSet('RUNNING','PASSED','FAILED','SKIPPED','NOT_APPLICABLE')][string]$Status,[string]$Message='')
    $Stages[$Stage] = [ordered]@{ status=$Status; message=$Message; updated_at=(Get-Date).ToString('o') }
    $state = [ordered]@{
        project_name=$ProjectName; run_id=$RunId; mode=$Mode; repository=$Repo;
        output_root=$OutputRoot; latest_stage=$Stage; stages=$Stages; updated_at=(Get-Date).ToString('o')
    }
    $tmp = "$StatePath.tmp"
    $state | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $tmp -Encoding utf8
    Move-Item -LiteralPath $tmp -Destination $StatePath -Force
}

function Assert-Path {
    param([string]$Stage,[string]$Path,[ValidateSet('Any','File','Directory')][string]$Kind='Any')
    if (-not (Test-Path -LiteralPath $Path)) { throw "[$Stage] Missing path: $Path" }
    if ($Kind -eq 'File' -and -not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "[$Stage] Expected file: $Path" }
    if ($Kind -eq 'Directory' -and -not (Test-Path -LiteralPath $Path -PathType Container)) { throw "[$Stage] Expected directory: $Path" }
}

function Test-Selected {
    param([string]$Target)
    return ($Mode -eq 'ALL' -or $Mode -eq $Target)
}

function Get-Sha256 {
    param([string]$Path)
    Assert-Path -Stage $CurrentStage -Path $Path -Kind File
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-NativeChecked {
    param(
        [string]$Stage,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )

    if ([System.IO.Path]::IsPathRooted($FilePath)) {
        Assert-Path -Stage $Stage -Path $FilePath -Kind File
    } elseif (-not (Get-Command $FilePath -ErrorAction SilentlyContinue)) {
        throw "[$Stage] Command not found: $FilePath"
    }
    Assert-Path -Stage $Stage -Path $WorkingDirectory -Kind Directory

    $log = Join-Path $LogRoot "$Stage.log"
    $stdout = Join-Path $LogRoot "$Stage.stdout.tmp"
    $stderr = Join-Path $LogRoot "$Stage.stderr.tmp"
    Remove-Item -LiteralPath $stdout,$stderr -Force -ErrorAction SilentlyContinue

    $shown = "$FilePath " + ($ArgumentList -join ' ')
    Write-Stage $Stage "RUN: $shown"
    Save-State -Stage $Stage -Status RUNNING -Message $shown

    $before = Get-Location
    $previousErrorActionPreference = $ErrorActionPreference
    $exitCode = $null

    try {
        Set-Location -LiteralPath $WorkingDirectory

        # Windows PowerShell 5.1 can surface native stderr as ErrorRecord objects.
        # Keep stderr redirected, and do not let informational stderr terminate
        # the controller. Success/failure is determined solely by exit code below.
        $ErrorActionPreference = 'SilentlyContinue'

        & $FilePath @ArgumentList 1> $stdout 2> $stderr
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Set-Location -LiteralPath $before.Path
    }

    foreach ($streamFile in @($stdout,$stderr)) {
        if (Test-Path -LiteralPath $streamFile -PathType Leaf) {
            Get-Content -LiteralPath $streamFile | ForEach-Object {
                Write-Host $_
                Add-Content -LiteralPath $log -Value $_ -Encoding utf8
            }
        }
    }
    Remove-Item -LiteralPath $stdout,$stderr -Force -ErrorAction SilentlyContinue

    if ($exitCode -ne 0) { throw "[$Stage] Exit code $exitCode. Full log: $log" }
}

function Invoke-PythonProbe {
    param(
        [string]$Stage,
        [string]$PythonPath,
        [string]$Name,
        [string]$Code,
        [string]$WorkingDirectory
    )
    # Avoid `python -c ...` from a generic PowerShell wrapper. Windows PowerShell
    # can reparse embedded quotes in a native command line. A real .py file has
    # no such argument-quoting ambiguity.
    $safe = ($Name -replace '[^A-Za-z0-9_.-]','_')
    $probe = Join-Path $ProbeRoot "probe-$safe.py"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($probe,$Code,$utf8NoBom)
    Invoke-NativeChecked -Stage $Stage -FilePath $PythonPath -ArgumentList @($probe) -WorkingDirectory $WorkingDirectory
}

function Resolve-Python {
    $candidates = @(
        @{ cmd='py'; args=@('-3.11') },
        @{ cmd='py'; args=@('-3.12') },
        @{ cmd='py'; args=@('-3.13') },
        @{ cmd='python'; args=@() }
    )
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.cmd -ErrorAction SilentlyContinue)) { continue }
        & $candidate.cmd @($candidate.args + @('--version')) 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return [ordered]@{ command=[string]$candidate.cmd; args=[string[]]$candidate.args }
        }
    }
    throw 'Python 3.11, 3.12, or 3.13 was not found.'
}

function Apply-ChronosWarningFix {
    param([string]$Stage)
    $target = Join-Path $Repo 'src\inventory_ai\models\chronos_adapter.py'
    Assert-Path -Stage $Stage -Path $target -Kind File
    $legacy = 'output["date_idx"] = ((output["timestamp"] - pd.Timestamp("2000-01-01")) / pd.Timedelta(days=1)).astype(int)'
    $fixed = 'output["date_idx"] = (output["timestamp"] - pd.Timestamp("2000-01-01")).dt.days.astype(int)'
    $text = Get-Content -LiteralPath $target -Raw
    if ($text.Contains($fixed)) { Write-Stage $Stage 'Chronos date-index implementation already fixed.'; return }
    if (-not $text.Contains($legacy)) { throw "[$Stage] Chronos adapter source differs from verified contract; automatic patch refused." }
    $backup = Join-Path $PatchRoot 'chronos_adapter.py.before.patch'
    Copy-Item -LiteralPath $target -Destination $backup -Force
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($target,$text.Replace($legacy,$fixed),$utf8NoBom)
    [ordered]@{ file=$target; backup=$backup; original_sha256=(Get-Sha256 $backup); patched_sha256=(Get-Sha256 $target) } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PatchRoot 'chronos_patch.json') -Encoding utf8
    Write-Stage $Stage 'Applied verified Chronos date-index warning fix; backup and hashes recorded outside repository.'
}

function Resolve-M5Zip {
    param([string]$InputPath)
    Assert-Path -Stage 'P06_DATA' -Path $InputPath -Kind Any
    if (Test-Path -LiteralPath $InputPath -PathType Leaf) {
        if ($InputPath.EndsWith('.zip',[System.StringComparison]::OrdinalIgnoreCase)) { return (Resolve-Path -LiteralPath $InputPath).Path }
        throw '[P06_DATA] M5 input file must be a ZIP.'
    }
    $direct = Join-Path $InputPath 'm5-forecasting-accuracy.zip'
    if (Test-Path -LiteralPath $direct -PathType Leaf) { return (Resolve-Path -LiteralPath $direct).Path }
    $required = @('sales_train_evaluation.csv','calendar.csv','sell_prices.csv')
    $dirs = @($InputPath) + @(Get-ChildItem -LiteralPath $InputPath -Directory -Recurse -ErrorAction Stop | ForEach-Object FullName)
    foreach ($dir in $dirs) {
        if (@($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $dir $_) -PathType Leaf) }).Count -eq 0) {
            $cache = Join-Path $RunRoot 'cache\m5-forecasting-accuracy.windows-cache.zip'
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $cache) | Out-Null
            if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) {
                Push-Location -LiteralPath $dir
                try { Compress-Archive -LiteralPath $required -DestinationPath $cache -Force }
                finally { Pop-Location }
            }
            return $cache
        }
    }
    throw "[P06_DATA] M5 ZIP or required CSV files were not found under: $InputPath"
}

function Assert-RunArtifacts {
    param([string]$Stage,[string]$ConfigPath,[string]$ExpectedRunName)
    $env:AMAZON08_REPO = $Repo
    $env:AMAZON08_CONFIG = $ConfigPath
    $env:AMAZON08_RUN = $ExpectedRunName
    $code = @'
import json, os
from pathlib import Path
from inventory_ai.config import load_config
root = Path(os.environ["AMAZON08_REPO"])
cfg = load_config(Path(os.environ["AMAZON08_CONFIG"]))
if cfg.run_name != os.environ["AMAZON08_RUN"]:
    raise SystemExit(f"unexpected run name: {cfg.run_name}")
report = root / cfg.output_dir
artifact = root / cfg.artifact_dir
for path in (report / "release_gate.json", report / "metrics_summary.json", artifact / "run_manifest.json"):
    if not path.exists(): raise SystemExit(f"missing output: {path}")
gate = json.loads((report / "release_gate.json").read_text(encoding="utf-8"))
metrics = json.loads((report / "metrics_summary.json").read_text(encoding="utf-8"))
if gate.get("gate_status") != "PASS": raise SystemExit(f"release gate: {gate.get('gate_status')}")
print(json.dumps({"run_name":cfg.run_name,"gate_status":gate.get("gate_status"),"selected_model":metrics.get("selected_model"),"candidate_wape_improvement":metrics.get("candidate_wape_improvement"),"candidate_cost_regression":metrics.get("candidate_cost_regression")},sort_keys=True))
'@
    try { Invoke-PythonProbe -Stage $Stage -PythonPath $VenvPython -Name "artifacts-$ExpectedRunName" -Code $code -WorkingDirectory $Repo }
    finally { Remove-Item Env:AMAZON08_REPO,Env:AMAZON08_CONFIG,Env:AMAZON08_RUN -ErrorAction SilentlyContinue }
}

try {
    # P00 — preflight
    $CurrentStage = 'P00_PREFLIGHT'
    Write-Stage $CurrentStage 'Starting repo, Python, disk, and GPU preflight.'
    Save-State -Stage $CurrentStage -Status RUNNING
    Assert-Path -Stage $CurrentStage -Path $Repo -Kind Directory
    foreach ($relative in @('pyproject.toml','requirements.txt','constraints\base.txt','configs\smoke.yaml','configs\m5_smoke.yaml','scripts\run_runtime_smoke.py','scripts\run_pipeline.py','scripts\run_sql_marts.py','scripts\package_release.py','scripts\verify_archive.py')) {
        Assert-Path -Stage $CurrentStage -Path (Join-Path $Repo $relative) -Kind File
    }
    Apply-ChronosWarningFix -Stage $CurrentStage
    $selected = Resolve-Python
    $PythonLauncher = $selected.command
    $PythonPrefix = $selected.args
    & $PythonLauncher @($PythonPrefix + @('--version')) 2>&1 | Set-Content -LiteralPath (Join-Path $RunRoot 'python_version.txt') -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw '[P00_PREFLIGHT] Python version query failed.' }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { & nvidia-smi 2>&1 | Set-Content -LiteralPath (Join-Path $RunRoot 'gpu_info.txt') -Encoding utf8 }
    Get-PSDrive -PSProvider FileSystem | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunRoot 'disk_info.json') -Encoding utf8
    Save-State -Stage $CurrentStage -Status PASSED

    # P03 — venv
    $CurrentStage = 'P03_VENV'
    $Venv = Join-Path $Repo '.venv'
    $VenvPython = Join-Path $Venv 'Scripts\python.exe'
    if ((Test-Path -LiteralPath $Venv -PathType Container) -and -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw "[$CurrentStage] Existing .venv is incomplete. It was not changed; inspect it manually."
    }
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $PythonLauncher -ArgumentList @($PythonPrefix + @('-m','venv',$Venv)) -WorkingDirectory $Repo
    }
    $versionCode = @'
import sys
print(sys.executable)
print(sys.version)
raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,14) else 7)
'@
    Invoke-PythonProbe -Stage $CurrentStage -PythonPath $VenvPython -Name 'venv-version' -Code $versionCode -WorkingDirectory $Repo
    Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('-m','pip','install','--upgrade','pip') -WorkingDirectory $Repo
    Save-State -Stage $CurrentStage -Status PASSED

    # P04 — dependencies
    $CurrentStage = 'P04_DEPENDENCIES'
    Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('-m','pip','install','-r','requirements.txt','--report',(Join-Path $ReportRoot 'pip_install_report.json')) -WorkingDirectory $Repo
    Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('-m','pip','check') -WorkingDirectory $Repo
    & $VenvPython -m pip inspect | Set-Content -LiteralPath (Join-Path $ReportRoot 'pip_inspect.json') -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "[$CurrentStage] pip inspect failed." }
    & $VenvPython -m pip freeze --all | Set-Content -LiteralPath (Join-Path $ReportRoot 'pip_freeze_all.txt') -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "[$CurrentStage] pip freeze failed." }
    $packageCode = @'
import importlib.metadata as md
import inventory_ai
print("module=", inventory_ai.__file__)
print("package_version=", inventory_ai.__version__)
print("distribution_version=", md.version("state-aligned-stockout-inventory-system"))
'@
    Invoke-PythonProbe -Stage $CurrentStage -PythonPath $VenvPython -Name 'installed-package' -Code $packageCode -WorkingDirectory $Repo
    $InventoryCli = Join-Path $Venv 'Scripts\inventory-ai.exe'
    Invoke-NativeChecked -Stage $CurrentStage -FilePath $InventoryCli -ArgumentList @('--version') -WorkingDirectory $Repo
    Invoke-NativeChecked -Stage $CurrentStage -FilePath $InventoryCli -ArgumentList @('--help') -WorkingDirectory $Repo
    Save-State -Stage $CurrentStage -Status PASSED

    $CurrentStage = 'P05_GPU'
    Write-Stage $CurrentStage 'CPU core only. PyTorch/CUDA not installed because it is optional for this repo.'
    Save-State -Stage $CurrentStage -Status NOT_APPLICABLE -Message 'GPU optional.'

    if ($Mode -eq 'SETUP') {
        $CurrentStage = 'P14_FINALIZE'; Save-State -Stage $CurrentStage -Status PASSED -Message 'SETUP COMPLETE'
        Write-Host "SETUP COMPLETE. State: $StatePath"; exit 0
    }

    # P06/P09 — M5 only if requested
    if (Test-Selected 'M5') {
        $CurrentStage = 'P06_DATA'

        if ([string]::IsNullOrWhiteSpace($M5Input)) {
            throw '[P06_DATA] M5 input is required for Mode=ALL or Mode=M5. Provide -M5Input <M5 ZIP or extracted dataset directory>, or set AMAZON08_M5_INPUT.'
        }

        $M5Zip = Resolve-M5Zip -InputPath $M5Input
        [ordered]@{ path=$M5Zip; bytes=(Get-Item -LiteralPath $M5Zip).Length; sha256=(Get-Sha256 $M5Zip) } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunRoot 'm5_fingerprint.json') -Encoding utf8
        $LocalM5Config = Join-Path $ConfigRoot 'm5_smoke.windows.yaml'
        $yamlPath = $M5Zip.Replace('\','/').Replace("'","''")
        $base = Get-Content -LiteralPath (Join-Path $Repo 'configs\m5_smoke.yaml') -Raw
        if (-not [regex]::IsMatch($base,'(?m)^\s*m5_zip_path:\s*.*$')) { throw '[P06_DATA] m5_zip_path missing in base config.' }
        [regex]::Replace($base,'(?m)^\s*m5_zip_path:\s*.*$',"  m5_zip_path: '$yamlPath'") | Set-Content -LiteralPath $LocalM5Config -Encoding utf8
        $env:AMAZON08_M5_ZIP = $M5Zip
        $env:AMAZON08_M5_CONFIG = $LocalM5Config
        $m5Code = @'
import os
from inventory_ai.contracts import validate_panel
from inventory_ai.data.m5 import load_m5_sample
from inventory_ai.config import load_config
frame = load_m5_sample(os.environ["AMAZON08_M5_ZIP"], n_series=8, history_days=70)
result = validate_panel(frame)
cfg = load_config(os.environ["AMAZON08_M5_CONFIG"])
print({"rows":len(frame),"series":result.series,"run_name":cfg.run_name,"m5_zip_path":cfg.data.m5_zip_path})
'@
        try { Invoke-PythonProbe -Stage $CurrentStage -PythonPath $VenvPython -Name 'm5-adapter-and-config' -Code $m5Code -WorkingDirectory $Repo }
        finally { Remove-Item Env:AMAZON08_M5_ZIP,Env:AMAZON08_M5_CONFIG -ErrorAction SilentlyContinue }
        Save-State -Stage $CurrentStage -Status PASSED
    } else { Save-State -Stage 'P06_DATA' -Status SKIPPED -Message "Mode=$Mode does not need M5." }

    if (Test-Selected 'TESTS') {
        $CurrentStage = 'P07_TESTS'
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('-m','pytest','-q') -WorkingDirectory $Repo
        Save-State -Stage $CurrentStage -Status PASSED
    } else { Save-State -Stage 'P07_TESTS' -Status SKIPPED -Message "Mode=$Mode" }

    if (Test-Selected 'SYNTHETIC') {
        $CurrentStage = 'P08_SYNTHETIC'
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('scripts\run_runtime_smoke.py','--root',$Repo,'--config','configs\smoke.yaml') -WorkingDirectory $Repo
        Assert-RunArtifacts -Stage $CurrentStage -ConfigPath (Join-Path $Repo 'configs\smoke.yaml') -ExpectedRunName 'synthetic_smoke'
        Save-State -Stage $CurrentStage -Status PASSED
    } else { Save-State -Stage 'P08_SYNTHETIC' -Status SKIPPED -Message "Mode=$Mode" }

    if (Test-Selected 'M5') {
        $CurrentStage = 'P09_M5'
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('scripts\run_pipeline.py','--root',$Repo,'--config',$LocalM5Config) -WorkingDirectory $Repo
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('scripts\run_sql_marts.py','--root',$Repo,'--config',$LocalM5Config) -WorkingDirectory $Repo
        Assert-RunArtifacts -Stage $CurrentStage -ConfigPath $LocalM5Config -ExpectedRunName 'm5_smoke'
        Save-State -Stage $CurrentStage -Status PASSED
    } else { Save-State -Stage 'P09_M5' -Status SKIPPED -Message "Mode=$Mode" }

    if (Test-Selected 'PACKAGE') {
        $CurrentStage = 'P13_PACKAGE'
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('-m','build') -WorkingDirectory $Repo
        $PackageZip = Join-Path $Repo 'dist\state_aligned_stockout_inventory_system.zip'
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('scripts\package_release.py','--root',$Repo,'--output',$PackageZip) -WorkingDirectory $Repo
        $sidecar = "$PackageZip.sha256"
        Assert-Path -Stage $CurrentStage -Path $sidecar -Kind File
        $expected = ((Get-Content -LiteralPath $sidecar -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
        if ((Get-Sha256 $PackageZip) -ne $expected) { throw "[$CurrentStage] ZIP SHA-256 sidecar mismatch." }
        foreach ($archiveMode in @('static','tests','runtime')) {
            Invoke-NativeChecked -Stage $CurrentStage -FilePath $VenvPython -ArgumentList @('scripts\verify_archive.py','--archive',$PackageZip,'--mode',$archiveMode) -WorkingDirectory $Repo
        }
        Save-State -Stage $CurrentStage -Status PASSED

        $CurrentStage = 'P13B_WHEEL'
        $WheelVenv = Join-Path $RunRoot 'wheel-smoke-venv'
        $WheelPython = Join-Path $WheelVenv 'Scripts\python.exe'
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $PythonLauncher -ArgumentList @($PythonPrefix + @('-m','venv',$WheelVenv)) -WorkingDirectory $Repo
        $wheel = Get-ChildItem -LiteralPath (Join-Path $Repo 'dist') -Filter 'state_aligned_stockout_inventory_system-*.whl' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $wheel) { throw "[$CurrentStage] Wheel not found." }
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $WheelPython -ArgumentList @('-m','pip','install',$wheel.FullName) -WorkingDirectory $env:TEMP
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $WheelPython -ArgumentList @('-m','pip','check') -WorkingDirectory $env:TEMP
        $wheelCode = @'
import importlib.metadata as md
import inventory_ai
print(inventory_ai.__version__)
print(md.version("state-aligned-stockout-inventory-system"))
'@
        Invoke-PythonProbe -Stage $CurrentStage -PythonPath $WheelPython -Name 'wheel-package' -Code $wheelCode -WorkingDirectory $env:TEMP
        $WheelCli = Join-Path $WheelVenv 'Scripts\inventory-ai.exe'
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $WheelCli -ArgumentList @('--version') -WorkingDirectory $env:TEMP
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $WheelCli -ArgumentList @('--help') -WorkingDirectory $env:TEMP
        Invoke-NativeChecked -Stage $CurrentStage -FilePath $WheelCli -ArgumentList @('--root',$Repo,'--config','configs\smoke.yaml') -WorkingDirectory $env:TEMP
        Save-State -Stage $CurrentStage -Status PASSED
    } else {
        Save-State -Stage 'P13_PACKAGE' -Status SKIPPED -Message "Mode=$Mode"
        Save-State -Stage 'P13B_WHEEL' -Status SKIPPED -Message "Mode=$Mode"
    }

    $CurrentStage = 'P14_FINALIZE'
    [ordered]@{
        project_name=$ProjectName; run_id=$RunId; mode=$Mode; repository=$Repo; venv_python=$VenvPython;
        m5_zip=$M5Zip; m5_local_config=$LocalM5Config; package_zip=$PackageZip;
        package_sha256=if ($PackageZip -and (Test-Path -LiteralPath $PackageZip)) { Get-Sha256 $PackageZip } else { $null };
        logs=$LogRoot; state=$StatePath; completed_at=(Get-Date).ToString('o')
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $ReportRoot 'final_summary.json') -Encoding utf8
    Save-State -Stage $CurrentStage -Status PASSED -Message 'COMPLETED'
    Write-Host "AMAZON08 LOCAL CONTROLLER COMPLETED. Summary: $(Join-Path $ReportRoot 'final_summary.json')"
}
catch {
    $message = $_.Exception.Message
    try { Write-Stage $CurrentStage "FAILED: $message"; Save-State -Stage $CurrentStage -Status FAILED -Message $message }
    catch { Write-Host "[$CurrentStage] FAILED: $message" }
    Write-Host "Controller stopped at stage: $CurrentStage"
    Write-Host "State: $StatePath"
    Write-Host "Logs: $LogRoot"
    exit 1
}
