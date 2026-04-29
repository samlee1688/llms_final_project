param(
    [switch]$InstallDependencies = $true,
    [switch]$BuildDatasets = $true,
    [switch]$RunExperiments = $true,
    [string]$RAMDocsSplit = "test",
    [ValidateSet("both", "openai", "anthropic")]
    [string]$JudgeSuite = "both",
    [int]$JudgeRounds = 3,
    [int]$MaxAblationRounds = 3,
    [string]$OutputRoot = "results\full_run",
    [string]$DataRoot = "data\full",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$Preferred)

    if ($Preferred -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }

    $candidates = @(
        (Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"),
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        "python"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -eq "python") { return $candidate }
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }

    return "python"
}

$projectRoot = $PSScriptRoot
$python = Resolve-PythonExe -Preferred $PythonExe
Write-Host "Using Python: $python"
Write-Host "Project root: $projectRoot"

$ramdocsOut      = Join-Path $projectRoot "$DataRoot\ramdocs_$RAMDocsSplit.json"
$tcbRamdocsOut   = Join-Path $projectRoot "$DataRoot\tcb_ramdocs_$RAMDocsSplit.json"
$confactSource   = Join-Path $projectRoot "$DataRoot\confact\humc_processed_cleaned_final.json"
$tcbConfactOut   = Join-Path $projectRoot "$DataRoot\confact\tcb_humc_processed_cleaned_final.json"

if ($InstallDependencies) {
    Write-Host "Installing dependencies..."
    & $python -m pip install -r (Join-Path $projectRoot "requirements.txt")
}

if ($BuildDatasets) {
    Write-Host "Converting RAMDocs..."
    & $python -m experiments.convert_ramdocs --split $RAMDocsSplit --output $ramdocsOut

    Write-Host "Building TCB from RAMDocs..."
    & $python -m experiments.build_tcb --input $ramdocsOut --benchmark ramdocs --output $tcbRamdocsOut

    Write-Host "Building ConfACT TCB (uses dedicated builder with Anthropic rewriter)..."
    if (Test-Path $confactSource) {
        $confactDestDir = Split-Path $tcbConfactOut -Parent
        & bash (Join-Path $projectRoot "experiments\build_tcb_confact.sh") `
            --source $confactSource --dest-dir $confactDestDir --rewriter-provider anthropic
    } else {
        Write-Warning "ConfACT source not found at $confactSource — skipping TCB build."
        Write-Warning "Run experiments/clean_confact_v2.py first to generate the source file."
    }
}

if ($RunExperiments) {
    $runs = @(
        @{ Name = "ramdocs";     Path = $ramdocsOut;    Benchmark = "ramdocs" },
        @{ Name = "tcb_ramdocs"; Path = $tcbRamdocsOut; Benchmark = "tcb"     },
        @{ Name = "confact";     Path = $confactSource; Benchmark = "confact"  },
        @{ Name = "tcb_confact"; Path = $tcbConfactOut; Benchmark = "tcb"      }
    )

    foreach ($run in $runs) {
        if (-not (Test-Path $run.Path)) {
            Write-Warning "Dataset not found: $($run.Path) — skipping $($run.Name)"
            continue
        }
        Write-Host "Running experiments for $($run.Name)..."
        $outDir = Join-Path $projectRoot (Join-Path $OutputRoot $run.Name)
        $ablationRounds = 0..$MaxAblationRounds
        & $python -m experiments.run_experiment `
            --dataset $run.Path `
            --benchmark $run.Benchmark `
            --systems standard_rag madam_rag eajp_scorecard_only eajp `
            --judge-rounds $JudgeRounds `
            --reuse-madam-debate-for-eajp `
            --ablate-rounds-from-cache $ablationRounds `
            --judge-suite $JudgeSuite `
            --output $outDir
    }
}

Write-Host "Done."
