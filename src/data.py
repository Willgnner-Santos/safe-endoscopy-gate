# -*- coding: utf-8 -*-
"""Dataset and dataloaders for the NORMAL x ALTERED binary gate."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import config as C


def load_fold_df(fold: int, part: str) -> pd.DataFrame:
    """Loads a split and applies the ambiguous policy, setting the 'y' column."""
    df = pd.read_csv(C.SPLITS_DIR / f"fold_{fold}_{part}.csv")
    df = df.drop_duplicates(C.IMG_COL).reset_index(drop=True)
    # binary target: y = ALTERED
    if C.AMBIGUOUS_POLICY == "drop_neither":
        neither = (df["NORMAL"] == 0) & (df["ALTERADO"] == 0)
        df = df[~neither].reset_index(drop=True)
    df["y"] = df[C.TARGET].astype(int)
    return df


def compute_pos_weight(fold: int) -> float:
    df = load_fold_df(fold, "train")
    pos = int(df["y"].sum()); neg = len(df) - pos
    return neg / max(pos, 1)


# ---------------------------------------------------------------------------
class GastroDataset:
    """Lazy: only imports torch/PIL when instantiated (keeps imports light)."""
    def __init__(self, df: pd.DataFrame, train: bool):
        from torchvision import transforms
        self.df = df.reset_index(drop=True)
        self.train = train
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.85, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
                transforms.ToTensor(),
                transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        import torch
        r = self.df.iloc[i]
        img = Image.open(C.IMAGES_DIR / r[C.IMG_COL]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(float(r["y"]))
        return x, y, r[C.IMG_COL]


def make_loaders(fold: int, batch_size=None, num_workers=None):
    import torch
    from torch.utils.data import DataLoader
    bs = batch_size or C.BATCH_SIZE
    nw = C.NUM_WORKERS if num_workers is None else num_workers
    tr = GastroDataset(load_fold_df(fold, "train"), train=True)
    va = GastroDataset(load_fold_df(fold, "val"),   train=False)
    te = GastroDataset(load_fold_df(fold, "test"),  train=False)
    pin = torch.cuda.is_available()
    return (
        DataLoader(tr, bs, shuffle=True,  num_workers=nw, pin_memory=pin, drop_last=False),
        DataLoader(va, bs, shuffle=False, num_workers=nw, pin_memory=pin),
        DataLoader(te, bs, shuffle=False, num_workers=nw, pin_memory=pin),
    )
