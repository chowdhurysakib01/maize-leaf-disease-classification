RICELEAF-DMNET v2 — THREE-STAGE FINE-TUNING

WHAT THIS PACKAGE DOES
- Reuses the exact RiceLeaf-DMNet v2 architecture.
- Strictly loads the existing epoch-21 best_model.pth checkpoint.
- Stage 1: trains the classifier only.
- Stage 2: unfreezes MobileNetV3-Large's final four feature blocks, DenseNet121 denseblock4/norm5, both feature encoders, fusion gate and classifier.
- Stage 3: unfreezes the full network.
- Uses differential AdamW learning rates for backbone, fusion/encoders and classifier groups.
- Uses the existing cosine scheduler factory.
- Selects the best checkpoint using validation Macro-F1.
- Loads train.csv and validation.csv only. It does not open test.csv or test images.
- Checks the split fingerprint and training class counts against the source checkpoint.
- Keeps pretrained backbone BatchNorm running statistics fixed during fine-tuning.

INSTALL
1. Extract the package into the existing D:\MaizeLeaf project root, preserving folders.
2. Ensure the original project already contains Dataset\, data_splits\, configs\, Models\RiceLeaf-DMNet.py and the other pipeline modules.
3. Ensure the source checkpoint exists at:
   D:\MaizeLeaf\Results\riceleaf_dmnet\final_riceleaf_dmnet_v2_seed42\checkpoints\best_model.pth
4. Activate the same project virtual environment.
5. Run from D:\MaizeLeaf:
   powershell -ExecutionPolicy Bypass -File .\RUN_RiceLeaf-DMNet-Finetune.ps1

VALIDATE WITHOUT TRAINING
python -m src.engine.finetune_trainer --validate-only --checkpoint "D:\MaizeLeaf\Results\riceleaf_dmnet\final_riceleaf_dmnet_v2_seed42\checkpoints\best_model.pth"

OUTPUT
Results\riceleaf_dmnet_finetuned\RiceLeaf-DMNet-Finetuned_seed42\
  history.csv
  training_summary.json
  checkpoints\best_model.pth
  checkpoints\best_model_info.json
  checkpoints\last_checkpoint.pth

IMPORTANT
The fine-tuned model is the same architecture with updated learned weights, not a different architecture. The original epoch-21 checkpoint remains untouched. The best fine-tuned checkpoint is saved separately. No performance improvement is guaranteed; compare validation Macro-F1 with the reproduced baseline.
