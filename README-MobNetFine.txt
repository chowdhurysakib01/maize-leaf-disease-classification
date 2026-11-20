# MobNetFine — Complete Progressive MSAF Pipeline

This implementation adds **MobNetFine** to the existing Maize Leaf Classification project.

## Architecture

- Backbone: ImageNet-pretrained MobileNetV3-Large.
- MSAF: five feature scales from MobileNetV3-Large (`features[2]`, `[4]`, `[7]`, `[13]`, `[16]`).
- Each scale is projected to a common channel dimension and receives CBAM-style channel/spatial attention.
- A sample-wise softmax scale-attention module weights the five scales.
- The weighted scales are concatenated and fused using a 1x1 convolution.
- Global average, global max and 2x2 regional pooling feed the final classifier.
- Output: seven raw class logits.

## Progressive fine-tuning

The `--epochs` budget is allocated 15% / 25% / 60%:

- Stage 1 — classifier-only: train the new MSAF/head layers; freeze MobileNetV3.
- Stage 2 — progressive: unfreeze the last MobileNetV3 feature blocks (`features[13:]`) and train the MSAF/head.
- Stage 3 — full: unfreeze the complete MobileNetV3 feature extractor and train the MSAF/head.

Examples:
- `--epochs 20` -> 3 + 5 + 12 epochs.
- `--epochs 50` -> 8 + 12 + 30 epochs.

Differential learning rates are used:
- Stage 1 head: base `--lr`.
- Stage 2 backbone: 5% of base LR; head: 30%.
- Stage 3 backbone: 1% of base LR; head: 10%.

Backbone BatchNorm running statistics remain frozen to reduce instability with small batches. Early stopping is active only during Stage 3 and monitors the selected validation metric.

## Balanced objective

MobNetFine enforces inverse-frequency weighted cross-entropy. Class weights are calculated exclusively from `train.csv`. Weighted sampling is disabled to avoid applying class-imbalance correction twice.

## Install

1. Extract `MobNetFine_Implementation_Patch.zip` into `D:\MaizeLeaf`, preserving folders and allowing the listed files to merge/replace.
2. Keep the existing `Dataset\`, `data_splits\train.csv`, `data_splits\validation.csv`, `data_splits\test.csv`, and `data_splits\split_metadata.json`.
3. Activate the existing `.venv`.
4. Run the model smoke test:

   ```powershell
   python -m Models.MobNetFine
   ```

5. Validate the trainer CLI:

   ```powershell
   python -m src.engine.trainer --model mobnetfine --pretrained --epochs 50 --batch-size 8 --image-size 224 --lr 0.0003 --optimizer adamw --weight-decay 0.0001 --scheduler cosine --early-stopping --patience 7 --min-delta 0.0001 --monitor val_macro_f1 --monitor-mode max --class-weighting inverse_frequency --sampler none --device cuda --amp --amp-dtype float16 --num-workers 4 --seed 42 --run-name final_mobnetfine_seed42 --validate-only
   ```

6. Run the complete pipeline:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\RUN_MobNetFine-Complete.ps1
   ```

The first run may download the official torchvision ImageNet MobileNetV3-Large weights if they are not already cached.

## Output

`Results\mobnetfine\final_mobnetfine_seed42\`

The existing trainer pipeline creates:

- `checkpoints\best_model.pth` and `checkpoints\last_checkpoint.pth`
- `training_history\training_history.csv` and JSON history/logs
- `training_curves\` — training/validation loss, accuracy, Macro-F1 and learning-rate plots
- `evaluation\validation\` — classification metrics, confusion matrix, ROC/PR, calibration and per-image predictions
- `evaluation\validation\error_analysis\` — confusion/error analysis
- `benchmarking\` — parameters, MACs/FLOPs, latency, throughput and memory
- `Xai\gradcam\`, `Xai\gradcam_plus_plus\`, `Xai\lime\`
- `LLM\` — English and Bangla reports
- `mobnetfine_training_strategy.json`
- `experiment_metadata.json`, `run_summary.json`, `pipeline_summary.json`

## Research integrity

- The existing fixed split fingerprint is retained. Train/validation per-class counts are also checked against `split_metadata.json` before training and class-weight calculation.
- The test partition is not used for training, validation-based checkpoint selection, XAI or post-training evaluation.
- The split metadata in the supplied source archive describes an image-level stratified split and has no plant/session group IDs. Group-level leakage therefore cannot be independently ruled out; this is a limitation of the current dataset split, not something this model can fix.
- Existing model architectures and their result folders are not changed or deleted.


## Implementation smoke tests

The supplied implementation was checked before packaging:

- `python -m Models.MobNetFine` — forward/backward pass, seven-logit output, Grad-CAM target and all three progressive parameter stages passed.
- `python -m Models.model_factory` — all nine registered models and backbone-freezing checks passed.
- Existing Grad-CAM model reconstruction — MobNetFine checkpoint loaded strictly and target layer resolved.
- Existing efficiency-benchmark model reconstruction — MobNetFine checkpoint loaded successfully.
- `python -m src.engine.trainer --help` — `mobnetfine` appears in the CLI model choices.
- Configuration validation with the intended 50-epoch, batch-8 settings passed.

The model has approximately 3.73 million parameters in the smoke-test environment. These checks do not replace the actual Windows/CUDA training run; the complete training and post-training pipeline must still be executed in your `D:\MaizeLeaf` environment.
