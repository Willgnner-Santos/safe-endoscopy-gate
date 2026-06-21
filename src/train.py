# -*- coding: utf-8 -*-
"""Training loop with atomic checkpoint/resume and logging.

Resilience: at each epoch it saves 'last.ckpt' (full state) and, when it improves,
'best.ckpt'. If the process dies, train_fold() resumes from the last saved epoch.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import config as C
from src import utils, data as datamod, model as modelmod, metrics as M


def _evaluate(model, loader, device, criterion):
    import torch
    model.eval()
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device).unsqueeze(1)
            logit = model(x)
            losses.append(criterion(logit, y).item())
            ps.append(torch.sigmoid(logit).cpu().numpy().ravel())
            ys.append(y.cpu().numpy().ravel())
    y = np.concatenate(ys); p = np.concatenate(ps)
    disc = M.discrimination(y, p)
    return float(np.mean(losses)), disc, y, p


def train_fold(model_key: str, fold: int, resume: bool = True, max_epochs=None):
    import torch
    from torch import nn
    device = utils.get_device()
    utils.set_seed(C.SEEDS[fold])
    tag = utils.fold_tag(model_key, fold)
    log = utils.get_logger(tag, C.LOGS_DIR / f"{tag}.log")
    ckpt_last = C.CKPT_DIR / f"{tag}_last.ckpt"
    ckpt_best = C.CKPT_DIR / f"{tag}_best.ckpt"
    max_epochs = max_epochs or C.EPOCHS

    # micro-batch per model (VRAM) + accumulation up to effective batch C.BATCH_SIZE
    micro = C.MICRO_BATCH.get(model_key, C.DEFAULT_MICRO_BATCH)
    accum = max(1, C.BATCH_SIZE // micro)
    tr, va, te = datamod.make_loaders(fold, batch_size=micro)
    pos_weight = datamod.compute_pos_weight(fold)
    log.info(f"[{tag}] device={device} | train={len(tr.dataset)} val={len(va.dataset)} "
             f"test={len(te.dataset)} | pos_weight={pos_weight:.3f} | "
             f"micro_batch={micro} x accum={accum} -> effective={micro*accum}")

    model = modelmod.build_model(model_key).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(modelmod.param_groups(model), weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    scaler = torch.cuda.amp.GradScaler(enabled=C.AMP and device.type == "cuda")

    start_epoch, best_metric, bad = 0, -1.0, 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auprc": [], "val_auroc": []}

    # ---- RESUME ----
    if resume:
        st = utils.load_checkpoint(ckpt_last, map_location=device)
        if st is not None:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
            start_epoch = st["epoch"] + 1; best_metric = st["best_metric"]
            bad = st["bad"]; history = st["history"]
            log.info(f"[{tag}] RESUME from epoch {start_epoch} (best={best_metric:.4f})")

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(it, **k): return it

    timer = utils.Timer()
    for epoch in range(start_epoch, max_epochs):
        model.train(); tl = []
        pbar = tqdm(tr, desc=f"{tag} ep{epoch:02d}", leave=False)
        opt.zero_grad()
        n_batches = len(tr)
        for bi, (x, y, _) in enumerate(pbar):
            x = x.to(device); y = y.to(device).unsqueeze(1)
            with torch.cuda.amp.autocast(enabled=C.AMP and device.type == "cuda"):
                logit = model(x)
                loss = criterion(logit, y) / accum     # normalize for accumulation
            scaler.scale(loss).backward()
            # optimizer step every `accum` micro-batches (or on the last one)
            if (bi + 1) % accum == 0 or (bi + 1) == n_batches:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tl.append(loss.item() * accum)             # actual loss (undoes normalization)
            try: pbar.set_postfix(loss=f"{tl[-1]:.3f}")
            except Exception: pass
        train_loss = float(np.mean(tl))
        val_loss, disc, _, _ = _evaluate(model, va, device, criterion)
        sched.step(disc["auprc"])

        history["epoch"].append(epoch); history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss); history["val_auprc"].append(disc["auprc"])
        history["val_auroc"].append(disc["auroc"])
        log.info(f"[{tag}] ep {epoch:02d} | train_loss {train_loss:.4f} | "
                 f"val_loss {val_loss:.4f} | val AUPRC {disc['auprc']:.4f} AUROC {disc['auroc']:.4f} "
                 f"| {timer.hms()}")

        improved = disc[C.MONITOR] > best_metric + 1e-4
        if improved:
            best_metric = disc[C.MONITOR]; bad = 0
            utils.save_checkpoint({"model": model.state_dict(), "epoch": epoch,
                                   "best_metric": best_metric, "model_key": model_key,
                                   "fold": fold}, ckpt_best)
        else:
            bad += 1

        # 'last' always (for resume)
        utils.save_checkpoint({"model": model.state_dict(), "opt": opt.state_dict(),
                               "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                               "epoch": epoch, "best_metric": best_metric, "bad": bad,
                               "history": history, "model_key": model_key, "fold": fold},
                              ckpt_last)
        if bad >= C.PATIENCE:
            log.info(f"[{tag}] early stopping (no improvement for {bad} epochs)"); break

    # plot training curves
    try:
        from src import plots
        plots.plot_training(history, model_key, fold)
    except Exception as e:
        log.info(f"[{tag}] training plot failed: {e}")

    # ---- final evaluation on TEST with the best checkpoint ----
    best = utils.load_checkpoint(ckpt_best, map_location=device)
    if best is not None:
        model.load_state_dict(best["model"])
    _, _, y_te, p_te = _evaluate(model, te, device, criterion)
    out = {"model_key": model_key, "fold": fold, "best_val_metric": best_metric,
           "history": history,
           "test": M.discrimination(y_te, p_te)}
    utils.save_json(out, C.METRICS_DIR / f"{tag}_metrics.json")
    # saves predictions for posterior analysis (gate, calibration, miss-rate)
    np.savez(C.PRED_DIR / f"{tag}_preds.npz", y=y_te, p=p_te,
             image_name=_test_names(fold))
    log.info(f"[{tag}] END | test AUPRC {out['test']['auprc']:.4f} AUROC {out['test']['auroc']:.4f}")
    return out


def _test_names(fold):
    df = datamod.load_fold_df(fold, "test")
    return df[C.IMG_COL].to_numpy()
