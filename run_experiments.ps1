param(
    [switch]$InstallDependencies = $true,
    [switch]$BuildDatasets = $true,
    [switch]$RunExperiments = $true,
    [ValidateSet("validation", "test")]
    [string]$AmbigDocsSplit = "validation",
    [string]$FaithEvalSplit = "test",
    [ValidateSet("both", "openai", "anthropic")]
    [string]$JudgeSuite = "both",
    [string]$OutputRoot = "results\full_run",
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
        if ($candidate -eq "python") {
            return $candidate
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return "python"
}

$projectRoot = $PSScriptRoot
$python = Resolve-PythonExe -Preferred $PythonExe
Write-Host "Using Python: $python"
Write-Host "Project root: $projectRoot"

$ambigdocsOut = Join-Path $projectRoot "data\ambigdocs_$AmbigDocsSplit.json"
$faithevalOut = Join-Path $projectRoot "data\faitheval_combined.json"
$tcbAmbigOut = Join-Path $projectRoot "data\tcb_ambigdocs.json"
$tcbFaithOut = Join-Path $projectRoot "data\tcb_faitheval.json"

if ($InstallDependencies) {
    Write-Host "Installing dependencies..."
    & $python -m pip install -r (Join-Path $projectRoot "requirements.txt")
}

if ($BuildDatasets) {
    Write-Host "Converting AmbigDocs..."
    & $python -m experiments.convert_ambigdocs --split $AmbigDocsSplit --output $ambigdocsOut

    Write-Host "Converting FaithEval..."
    & $python -m experiments.convert_faitheval --dataset-types counterfactual inconsistent --split $FaithEvalSplit --output $faithevalOut

    Write-Host "Building TCB from AmbigDocs..."
    & $python -m experiments.build_tcb --input $ambigdocsOut --benchmark ambigdocs --output $tcbAmbigOut

    Write-Host "Building TCB from FaithEval..."
    & $python -m experiments.build_tcb --input $faithevalOut --benchmark faitheval --output $tcbFaithOut
}

if ($RunExperiments) {
    $runs = @(
        @{ Name = "ambigdocs"; Path = $ambigdocsOut; Benchmark = "ambigdocs" },
        @{ Name = "faitheval"; Path = $faithevalOut; Benchmark = "faitheval" },
        @{ Name = "tcb_ambigdocs"; Path = $tcbAmbigOut; Benchmark = "tcb" },
        @{ Name = "tcb_faitheval"; Path = $tcbFaithOut; Benchmark = "tcb" }
    )

    foreach ($run in $runs) {
        Write-Host "Running experiments for $($run.Name)..."
        $outDir = Join-Path $projectRoot (Join-Path $OutputRoot $run.Name)
        & $python -m experiments.run_experiment `
            --dataset $run.Path `
            --benchmark $run.Benchmark `
            --systems standard_rag madam_rag eajp `
            --judge-suite $JudgeSuite `
            --output $outDir
    }
}

Write-Host "Done."
