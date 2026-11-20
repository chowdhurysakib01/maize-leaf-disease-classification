$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$ProjectRoot = (Get-Location).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RunDirectory = Join-Path $ProjectRoot 'Results\mobnetfine\final_mobnetfine_seed42'

Write-Host ''
Write-Host 'MOBNETFINE - COMPLETE PROGRESSIVE MSAF TRAINING PIPELINE' -ForegroundColor Cyan
Write-Host 'This run uses ImageNet-pretrained MobileNetV3-Large and the MSAF head.'
Write-Host 'It will generate validation evaluation, benchmark, error analysis, Grad-CAM, Grad-CAM++, LIME and LLM reports.'
Write-Host ''

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project virtual environment not found: $Python"
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'Dataset') -PathType Container)) {
    throw 'Dataset folder not found.'
}

$RequiredFiles = @(
    'data_splits\split_metadata.json',
    'data_splits\train.csv',
    'data_splits\validation.csv',
    'data_splits\test.csv'
)

foreach ($RelativePath in $RequiredFiles) {
    $Path = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required fixed-split file not found: $Path"
    }
}

if (Test-Path -LiteralPath $RunDirectory) {
    Write-Host "Existing MobNetFine output will be replaced:" -ForegroundColor Yellow
    Write-Host $RunDirectory -ForegroundColor Yellow
    Write-Host 'No other model results, checkpoints, datasets or split files will be deleted.'
    $Confirmation = Read-Host 'Type YES to delete this MobNetFine run folder and start a fresh run'
    if ($Confirmation -cne 'YES') {
        Write-Host 'Cancelled. No files were deleted.' -ForegroundColor Yellow
        exit 1
    }
    Remove-Item -LiteralPath $RunDirectory -Recurse -Force
}

$TrainingArguments = @(
    '-m',
    'src.engine.trainer',
    '--model', 'mobnetfine',
    '--pretrained',
    '--no-freeze-backbone',
    '--epochs', '50',
    '--batch-size', '8',
    '--image-size', '224',
    '--lr', '0.0003',
    '--optimizer', 'adamw',
    '--weight-decay', '0.0001',
    '--scheduler', 'cosine',
    '--early-stopping',
    '--patience', '7',
    '--min-delta', '0.0001',
    '--monitor', 'val_macro_f1',
    '--monitor-mode', 'max',
    '--loss', 'cross_entropy',
    '--class-weighting', 'inverse_frequency',
    '--sampler', 'none',
    '--device', 'cuda',
    '--amp',
    '--amp-dtype', 'float16',
    '--gradient-accumulation-steps', '1',
    '--num-workers', '4',
    '--seed', '42',
    '--run-name', 'final_mobnetfine_seed42',
    '--run-evaluation',
    '--run-benchmark',
    '--run-error-analysis',
    '--run-xai',
    '--run-llm',
    '--save-last-checkpoint'
)

Write-Host ''
Write-Host 'Starting MobNetFine. ImageNet weights may be downloaded on the first run if not cached.' -ForegroundColor Cyan
& $Python @TrainingArguments

if ($LASTEXITCODE -ne 0) {
    throw "MobNetFine pipeline failed with exit code $LASTEXITCODE. Existing completed artifacts were preserved."
}

Write-Host ''
Write-Host 'MOBNETFINE COMPLETE PIPELINE FINISHED SUCCESSFULLY' -ForegroundColor Green
Write-Host "All results: $RunDirectory" -ForegroundColor Green
Write-Host "Best checkpoint: $(Join-Path $RunDirectory 'checkpoints\best_model.pth')" -ForegroundColor Green
Write-Host "Pipeline summary: $(Join-Path $RunDirectory 'pipeline_summary.json')" -ForegroundColor Green
Write-Host 'The held-out test split was not used.' -ForegroundColor Green
