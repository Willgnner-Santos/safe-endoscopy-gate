# -*- coding: utf-8 -*-
"""Central configuration for A4 — Binary Triage Gate (NORMAL x ALTERED).

Everything that changes between machines/environments stays HERE. Editable for local or Colab.
"""
from pathlib import Path
import os

# ----------------------------------------------------------------------------
# PATHS (adjust IMAGES_DIR according to where the images are: local or Drive)
# ----------------------------------------------------------------------------
# Root of the A4 project (this folder). Resolved automatically.
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Colab robustness: NB00 writes '.a4_env' (A4_IMAGES_DIR=...). If the env is not
# set in this session (kernel separated from NB00), we read this file automatically.
_envfile = PROJECT_DIR / ".a4_env"
if "A4_IMAGES_DIR" not in os.environ and _envfile.exists():
    for _ln in _envfile.read_text().splitlines():
        if _ln.startswith("A4_IMAGES_DIR=") and "=" in _ln:
            os.environ["A4_IMAGES_DIR"] = _ln.split("=", 1)[1].strip()

# .jpg images. Robust resolution: env A4_IMAGES_DIR takes priority; otherwise looks in
# the usual places (inside the project OR sibling to the project) and uses the 1st that exists.
def _resolve_images_dir():
    env = os.environ.get("A4_IMAGES_DIR")
    if env:
        return Path(env)
    candidates = [
        PROJECT_DIR / "Data" / "Imgs",          # inside the project (current)
        PROJECT_DIR.parent / "Data" / "Imgs",   # project sibling (old layout)
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # fallback (shows expected path even if missing)

IMAGES_DIR = _resolve_images_dir()

SPLITS_DIR     = PROJECT_DIR / "splits"
RESULTS_DIR    = PROJECT_DIR / "results"
LOGS_DIR       = RESULTS_DIR / "logs"
CKPT_DIR       = RESULTS_DIR / "checkpoints"
METRICS_DIR    = RESULTS_DIR / "metrics"
PRED_DIR       = RESULTS_DIR / "predictions"
FIG_DIR        = PROJECT_DIR / "figures"
FIG_EN, FIG_PT, FIG_TRAIN = FIG_DIR / "en", FIG_DIR / "pt", FIG_DIR / "training"

for d in [RESULTS_DIR, LOGS_DIR, CKPT_DIR, METRICS_DIR, PRED_DIR,
          FIG_DIR, FIG_EN, FIG_PT, FIG_TRAIN]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# DATA / BINARY TARGET
# ----------------------------------------------------------------------------
N_FOLDS = 5
IMG_COL = "image_name"
CENTER_COL = "LOCAL"                      # 1 -> 720 imgs, 2 -> 1270 imgs (confirmed)
TARGET = "ALTERADO"                       # positive target of the gate (1 = altered)
# The 7 pathologies (for safety/miss-rate analysis; NOT training targets)
PATHOLOGIES = ["ENANTEMA", "PÓLIPO", "ÚLCERA", "EROSÃO",
               "MICRONODULARIDADE", "ECTASIA VASCULAR", "NEOPLASIA"]
# Images "neither NORMAL nor ALTERED" (6 in splits) and "both" (4): treatment policy
#   "drop_neither": removes the 6 neither-nor; "both" -> ALTERED=1 (clinical conservative)
AMBIGUOUS_POLICY = "drop_neither"

# ----------------------------------------------------------------------------
# IMAGE / TRAINING
# ----------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# HARDWARE DETECTION (defines effective batch, micro-batch, and LR automatically)
# ----------------------------------------------------------------------------
# Philosophy: the same code runs on a 3 GB local GPU and on a 96 GB server GPU.
#   - Small GPU (<=8 GB): effective batch 32; heavy networks via micro-batch 8 + accumulation.
#   - Medium GPU (8-40 GB): effective batch 64; no accumulation (fits directly).
#   - Large GPU (>40 GB, e.g., H100 / RTX PRO 6000 Blackwell): effective batch 128; no accumulation.
# To force manually, set the env A4_BATCH_SIZE (e.g., '64').
def _detect_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0

_VRAM_GB = _detect_vram_gb()

def _auto_batch_size():
    forced = os.environ.get("A4_BATCH_SIZE")
    if forced:
        return int(forced)
    if _VRAM_GB > 40:   # H100, A100-80, RTX 6000 Blackwell
        return 128
    if _VRAM_GB > 8:    # medium cards (16-40 GB)
        return 64
    return 32           # small GPU (3-8 GB) or CPU

BATCH_SIZE = _auto_batch_size()  # EFFECTIVE batch (same for all models on the same hardware)

# NUM_WORKERS: auto-detect. 0 = no parallelism (debug/Windows). On servers with many cores,
# DataLoader uses more workers so as not to leave the GPU idle waiting for data.
import multiprocessing as _mp
NUM_WORKERS  = min(8, max(0, _mp.cpu_count() - 2))  # up to 8; on Windows, if it hangs, force 0
PIN_MEMORY   = True              # speeds up CPU->GPU transfer (disable if RAM error occurs)
EPOCHS       = 40
PATIENCE     = 8                 # early stopping (epochs without improvement in val AUPRC)

# --- VRAM: micro-batch per model ---
# micro-batch = what actually fits in VRAM at once; training accumulates gradients until
# BATCH_SIZE is reached. On a large GPU, micro = effective batch (accumulation = 1, no cost).
# On a small GPU, heavy networks use micro 8 and accumulate (effective batch preserved).
def _auto_micro_batch():
    models = ["mobilenetv3", "efficientnet_b0", "resnet50", "densenet121", "convnext_tiny"]
    if _VRAM_GB > 8:        # whole effective batch fits for all -> no accumulation
        return {k: BATCH_SIZE for k in models}
    # Small GPU/CPU: light ones fit 32, heavy ones use 8 (and accumulate up to 32)
    return {"mobilenetv3": 32, "efficientnet_b0": 32,
            "resnet50": 8, "densenet121": 8, "convnext_tiny": 8}

MICRO_BATCH = _auto_micro_batch()
DEFAULT_MICRO_BATCH = 8          # fallback for unlisted model

# LR scaled by batch size (square-root rule — gentle for fine-tuning).
# Base calibrated for batch 32. In batch 128 -> factor sqrt(4)=2x.
_LR_SCALE = (BATCH_SIZE / 32.0) ** 0.5
LR_BACKBONE  = 1e-4 * _LR_SCALE
LR_HEAD      = 1e-3 * _LR_SCALE
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.0
AMP          = True              # mixed precision (speeds up on GPU)
MONITOR      = "auprc"           # selection metric for the best checkpoint (val)
SEEDS        = [42, 43, 44, 45, 46]   # one seed per fold

# ----------------------------------------------------------------------------
# MODELS (timm) — 5 distinct families, light, with good pre-trained weights
# ----------------------------------------------------------------------------
MODELS = {
    "resnet50":      "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",   # in22k->in1k: better for small data
    "densenet121":   "densenet121.ra_in1k",
    "mobilenetv3":   "mobilenetv3_large_100.ra_in1k",
}
MODEL_FAMILY = {  # label for figures
    "resnet50": "CNN (residual)", "efficientnet_b0": "CNN (compound)",
    "convnext_tiny": "CNN (modern)", "densenet121": "CNN (dense)",
    "mobilenetv3": "CNN (mobile)",
}

# ----------------------------------------------------------------------------
# EVALUATION / GATE
# ----------------------------------------------------------------------------
# Clinical operating points: target-sensitivity and target-specificity
TARGET_SENS = [0.90, 0.95, 0.99]   # triage gate: high sensitivity
TARGET_SPEC = [0.90, 0.95]
BOOTSTRAP_N = 2000
ECE_BINS = 15

# ----------------------------------------------------------------------------
# FIGURES — bilingual texts (EN/PT). We ALWAYS plot both versions.
# ----------------------------------------------------------------------------
DPI = 200
LANGS = ("en", "pt")
