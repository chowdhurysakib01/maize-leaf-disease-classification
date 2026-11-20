# Maize Leaf Disease Classification

A deep learning research project focused on maize leaf disease and nutrient deficiency classification using conventional and hybrid neural network architectures.

## Overview

This project investigates deep learning approaches for automated maize leaf classification. It includes multiple CNN and transformer-based models, a proposed hybrid architecture (MaizeLeaf-DMNet), fine-tuning experiments, comprehensive performance evaluation, and Explainable AI (XAI).

The objective is to compare model performance, computational efficiency, generalization, and interpretability under consistent experimental conditions.

## Models

The research includes the following architectures:

- Custom CNN
- DenseNet121
- EfficientNet-B0
- MobileNetV3
- ResNet50
- Small Vision Transformer (Small ViT)
- MaizeLeaf-DMNet 
- maizeleaf_dmnet_finetuned 

## Experimental Setup

- Two experimental random seeds: **42 and 123**
- Consistent dataset split and preprocessing configurations
- Training and validation monitoring
- Model checkpoint selection
- Comparative performance evaluation

## Evaluation

The evaluation pipeline includes:

- Accuracy, Precision, Recall, and F1-score
- Classification reports and confusion matrices
- ROC-AUC and PR-AUC analysis
- Training and validation loss analysis
- Calibration and error analysis
- Model parameters and computational efficiency
- Training time and inference latency

## Explainable AI (XAI)

Model interpretability is investigated using:

- Grad-CAM
- Grad-CAM++
- LIME

The repository includes explainability reports, comparison results, and selected visual summaries.

## Repository Structure

text
MaizeLeaf/
├── configs/                 # Training configurations
├── data_splits/             # Dataset split files and metadata
├── heldout_test_source_bundle/
├── LLM/                     # Research analysis reports
├── Models/                  # Model implementations
├── Pipeline_Test/           # Pipeline verification
├── Results/                 # Seed 42 results
├── Results - 123/           # Seed 123 results
├── src/
│   ├── benchmarking/
│   ├── data/
│   ├── engine/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
└── RUN_*.ps1                # Execution scripts


## Installation

Requirements:

- Python 3.x
- PyTorch
- Torchvision
- Compatible CUDA environment (recommended for GPU training)

Create and activate a virtual environment:

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install PyTorch and the additional dependencies required by the project.

## Running the Pipeline

Example PowerShell commands:

powershell
powershell -ExecutionPolicy Bypass -File .\RUN_MaizeLeaf-DMNet-Complete.ps1

powershell
powershell -ExecutionPolicy Bypass -File .\RUN_Heldout-Test-Evaluation.ps1

Refer to the corresponding scripts and project configurations for execution details.

## Dataset and Checkpoints

The original dataset and trained model checkpoint files are not included in this repository. Users must obtain the dataset separately and configure the appropriate paths before running experiments.

## Research Scope

This repository supports comparative deep learning research, hybrid architecture evaluation, computational benchmarking, and explainability analysis for maize leaf classification.

## Repository

GitHub: https://github.com/chowdhurysakib01/maize-leaf-disease-classification

**Project:** Maize Leaf Disease Classification  
**Maintainer:** [@chowdhurysakib01](https://github.com/chowdhurysakib01)

