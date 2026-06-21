# Safe Endoscopy Gate: Binary Triage in Clinical Practice

This repository contains the official code for the paper evaluating a deep learning-based binary triage gate ("Normal" vs. "Altered") for upper gastrointestinal endoscopy. 

Unlike traditional studies that focus primarily on network architectures and ranking, this research is designed from a **clinical deployment perspective**. We emphasize:
- **Operating Points**: Prioritizing target-sensitivity for safe clinical routing.
- **Safety**: Detailed analysis of missed pathologies (false negatives) when the gate predicts "Normal".
- **Calibration**: Assessment of model reliability (ECE, Brier score) to ensure the predicted probabilities are clinically trustworthy.
- **Cascade Cost**: Evaluating the reduction in expert reader workload and the trade-offs of a two-stage cascade (AI pre-filter + human expert).

## Repository Structure

The source code is organized to ensure reproducibility and clarity for reviewers:

```text
├── configs/
│   └── config.py        # Centralized parameters (paths, models, training settings, metrics)
├── notebooks/           # Step-by-step pipeline from setup to results synthesis
│   ├── NB00_setup_colab.ipynb
│   ├── NB01_data_eda.ipynb
│   ├── NB02_train.ipynb
│   ├── NB03_operation_calibration.ipynb
│   ├── NB04_gate_safety_cascade.ipynb
│   └── NB05_synthesis_latex.ipynb
└── src/                 # Core implementation modules
    ├── data.py          # Data loading, ambiguous policy handling, and augmentations
    ├── model.py         # Model building (binary classification head)
    ├── metrics.py       # Comprehensive clinical metrics (AUROC, AUPRC, Net Benefit, Calibration)
    ├── train.py         # Training loop with atomic checkpointing and AMP
    ├── plots.py         # High-quality bilingual plotting utilities for publication
    └── utils.py         # Utilities for reproducibility (seeds, logging, I/O)
```

## Reproducibility

1. **Environment**: The models are built using `timm`, `torch`, and `torchvision`.
2. **Execution**: The pipeline runs sequentially from `NB00` to `NB05`. Training (`NB02_train.ipynb`) automatically scales micro-batch sizes based on available VRAM and supports atomic resuming from the last checkpoint to prevent data loss upon interruption.
3. **Data Splits**: 5-fold cross-validation splits are pre-defined to ensure robust and reproducible reporting of all metrics across all models.

## Key Outcomes

The final output of the pipeline (`NB05`) provides ready-to-use LaTeX tables and a compiled JSON with all evaluated metrics, enabling transparent review of the model's discrimination, calibration, net benefit, and its clinical viability as a workload-reduction gate.
