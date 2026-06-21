# -*- coding: utf-8 -*-
"""Utilities: seeds, device, logging, JSON, checkpoint/resume."""
import os, sys, json, random, logging, time
from pathlib import Path
import numpy as np


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        # deterministic but without killing performance
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str, log_file: Path):
    log_file = Path(log_file); log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def save_json(obj, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# CHECKPOINT / RESUME  — so the PC can stop and not lose anything
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, ckpt_path: Path):
    import torch
    ckpt_path = Path(ckpt_path); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, ckpt_path)   # atomic write: never corrupts the previous ckpt


def load_checkpoint(ckpt_path: Path, map_location=None):
    import torch
    if not Path(ckpt_path).exists():
        return None
    return torch.load(ckpt_path, map_location=map_location, weights_only=False)


def fold_tag(model_key: str, fold: int) -> str:
    return f"{model_key}_fold{fold}"


def is_run_done(metrics_dir: Path, model_key: str, fold: int) -> bool:
    """Is there already a final metric saved for (model, fold)? Allows skipping completed runs."""
    return (Path(metrics_dir) / f"{fold_tag(model_key, fold)}_metrics.json").exists()


class Timer:
    def __init__(self): self.t0 = time.time()
    def lap(self): return time.time() - self.t0
    def hms(self):
        s = int(self.lap()); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
