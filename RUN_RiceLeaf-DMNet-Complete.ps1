$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$ProjectRoot = (Get-Location).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$SourceCheckpoint = Join-Path $ProjectRoot 'Results\riceleaf_dmnet\final_riceleaf_dmnet_v2_seed42\checkpoints\best_model.pth'
$TrainingRun = Join-Path $ProjectRoot 'Results\riceleaf_dmnet_finetuned\RiceLeaf-DMNet-Finetuned_seed42'
$FinalRun = Join-Path $ProjectRoot 'Results\riceleaf_dmnet\finetuned_eval_seed42'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project virtual environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $SourceCheckpoint -PathType Leaf)) {
    throw "Original RiceLeaf-DMNet v2 best checkpoint not found: $SourceCheckpoint"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'data_splits\split_metadata.json') -PathType Leaf)) {
    throw 'Fixed split metadata not found. Restore data_splits\split_metadata.json before running.'
}

Write-Host ''
Write-Host 'RICELEAF-DMNET: COMPLETE FINE-TUNING + EVALUATION PIPELINE' -ForegroundColor Cyan
Write-Host 'This run will:'
Write-Host '  1. Remove only the previous fine-tuning output folder and previous fine-tuned evaluation folder.'
Write-Host '  2. Keep the original RiceLeaf-DMNet v2 checkpoint and all dataset/split files untouched.'
Write-Host '  3. Fine-tune from the original best checkpoint.'
Write-Host '  4. Generate classification, confusion matrix, ROC/PR, calibration, error analysis, Grad-CAM, Grad-CAM++, LIME and benchmark results.'
Write-Host '  5. Collect training history, checkpoints and curves into one final result folder.'
Write-Host ''
Write-Host "Original checkpoint (will NOT be deleted): $SourceCheckpoint" -ForegroundColor Green
Write-Host "Old fine-tuning output to be replaced:       $TrainingRun" -ForegroundColor Yellow
Write-Host "Old evaluation output to be replaced:       $FinalRun" -ForegroundColor Yellow
Write-Host ''
$confirmation = Read-Host 'Type YES to delete only the two old output folders and start a fresh complete run'
if ($confirmation -cne 'YES') {
    Write-Host 'Cancelled. No files were deleted.' -ForegroundColor Yellow
    exit 1
}

foreach ($target in @($TrainingRun, $FinalRun)) {
    if (Test-Path -LiteralPath $target) {
        Write-Host "Removing previous output: $target" -ForegroundColor Yellow
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

$TrainingParent = Split-Path -Parent $TrainingRun
if (-not (Test-Path -LiteralPath $TrainingParent)) {
    New-Item -ItemType Directory -Path $TrainingParent -Force | Out-Null
}

Write-Host ''
Write-Host '[1/3] STARTING THREE-STAGE FINE-TUNING' -ForegroundColor Cyan
& $Python -m src.engine.finetune_trainer `
    --checkpoint $SourceCheckpoint `
    --results-dir $TrainingParent `
    --run-name 'RiceLeaf-DMNet-Finetuned_seed42' `
    --device cuda `
    --batch-size 4 `
    --accumulation-steps 2 `
    --num-workers 4 `
    --epochs-stage1 3 `
    --epochs-stage2 6 `
    --epochs-stage3 12 `
    --amp `
    --amp-dtype float16 `
    --seed 42
if ($LASTEXITCODE -ne 0) {
    throw "Fine-tuning failed with exit code $LASTEXITCODE. Training artifacts have been kept at: $TrainingRun"
}

$FineTunedCheckpoint = Join-Path $TrainingRun 'checkpoints\best_model.pth'
if (-not (Test-Path -LiteralPath $FineTunedCheckpoint -PathType Leaf)) {
    throw "Fine-tuning finished without producing best_model.pth: $FineTunedCheckpoint"
}

Write-Host ''
Write-Host '[2/3] RUNNING COMPLETE VALIDATION EVALUATION' -ForegroundColor Cyan
& $Python -m src.evaluation.evaluate_finetuned_dmnet `
    --checkpoint $FineTunedCheckpoint `
    --run-dir $FinalRun `
    --device cuda `
    --batch-size 8 `
    --num-workers 0 `
    --samples-per-class 5 `
    --lime-num-samples 1000 `
    --lime-num-features 10 `
    --lime-num-segments 80 `
    --benchmark-warmup 5 `
    --benchmark-iterations 20
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE. Fine-tuning checkpoint and history remain safe at: $TrainingRun"
}

Write-Host ''
Write-Host '[3/3] COLLECTING ALL RESULTS INTO ONE FINAL RUN FOLDER' -ForegroundColor Cyan
& $Python -m src.evaluation.finetune_history_report `
    --training-dir $TrainingRun `
    --final-run-dir $FinalRun
if ($LASTEXITCODE -ne 0) {
    throw "Result collection failed with exit code $LASTEXITCODE. Original training artifacts remain safe at: $TrainingRun"
}

# Delete the temporary fine-tuning staging folder only after every evaluation and copy succeeds.
if (Test-Path -LiteralPath $TrainingRun) {
    Remove-Item -LiteralPath $TrainingRun -Recurse -Force
}

Write-Host ''
Write-Host 'COMPLETE PIPELINE FINISHED SUCCESSFULLY' -ForegroundColor Green
Write-Host "All final results: $FinalRun" -ForegroundColor Green
Write-Host "Combined summary:  $(Join-Path $FinalRun 'complete_run_summary.json')" -ForegroundColor Green
Write-Host 'The original RiceLeaf-DMNet v2 checkpoint and test split were not modified or accessed.' -ForegroundColor Green
