$ErrorActionPreference = "Stop"
Set-Location "D:\MaizeLeaf"

if (!(Test-Path ".\src\evaluation\heldout_test_evaluator.py")) {
    throw "Missing src\evaluation\heldout_test_evaluator.py. Copy it from the bundle first."
}

Write-Host "MAIZE LEAF - HELD-OUT TEST EVALUATION" -ForegroundColor Cyan
Write-Host "Output: D:\MaizeLeaf\Results\heldout_test_evaluation"
Write-Host "This does not train models or modify source checkpoints."
Write-Host ""

python -m src.evaluation.heldout_test_evaluator `
    --device cuda `
    --batch-size 8 `
    --num-workers 0 `
    --samples-per-class 5 `
    --lime-num-samples 1000 `
    --lime-num-features 10 `
    --lime-num-segments 80

if ($LASTEXITCODE -ne 0) {
    throw "Held-out test evaluation stopped with exit code $LASTEXITCODE."
}
