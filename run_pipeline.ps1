# Runs the full pipeline. Usage:  .\run_pipeline.ps1               (everything)
#                                 .\run_pipeline.ps1 -SkipTrain    (reuse saved models)
param([switch]$SkipTrain)

function Step($cmd) {
    Write-Host "==> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $cmd" }
}

Step "python -m src.data.generate_patterns"
Step "python -m src.data.build_dataset"
if (-not $SkipTrain) { Step "python -m src.train" }
Step "python -m src.evaluate"
Step "python -m scripts.make_example_payload"
Step "python -m pytest -q"
Write-Host "Pipeline finished." -ForegroundColor Green