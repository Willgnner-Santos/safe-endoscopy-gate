# -*- coding: utf-8 -*-
"""Metrics for the binary triage gate.

Clinical focus (not just F1/acc):
  - discrimination: AUROC, AUPRC, balanced acc, F1, MCC
  - operation: sensitivity@target-specificity and vice versa, Youden's J
  - calibration: ECE, Brier, MCE
  - gate safety: pathology miss-rate among predicted NORMAL
  - clinical decision: net benefit (decision curve)
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, balanced_accuracy_score,
                             matthews_corrcoef, confusion_matrix,
                             roc_curve, precision_recall_curve, brier_score_loss)


# ---------------------------------------------------------------------------
def discrimination(y, p):
    """Threshold-independent metrics."""
    y = np.asarray(y); p = np.asarray(p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "prevalence": float(y.mean()),
        "n": int(len(y)),
    }


def at_threshold(y, p, thr):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)         # recall / sensitivity
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "threshold": float(thr),
        "sensitivity": float(sens), "specificity": float(spec),
        "ppv": float(ppv), "npv": float(npv),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def youden_threshold(y, p):
    """Threshold that maximizes Youden's J (sens+spec-1)."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def threshold_for_sensitivity(y, p, target_sens):
    """Lowest threshold that guarantees sensitivity >= target (triage gate)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target_sens)[0]
    if len(ok) == 0:
        return 0.0
    # among those that achieve the sensitivity, pick the one with highest specificity (lowest fpr)
    idx = ok[np.argmin(fpr[ok])]
    return float(thr[idx])


def threshold_for_specificity(y, p, target_spec):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where((1 - fpr) >= target_spec)[0]
    if len(ok) == 0:
        return 1.0
    idx = ok[np.argmax(tpr[ok])]
    return float(thr[idx])


# ---------------------------------------------------------------------------
def calibration(y, p, n_bins=15):
    """ECE, MCE and Brier (calibration — critical for a clinical gate)."""
    y = np.asarray(y); p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    rows = []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.sum() == 0:
            rows.append({"bin": i, "conf": None, "acc": None, "count": 0}); continue
        conf = float(p[m].mean()); acc = float(y[m].mean()); w = m.mean()
        gap = abs(conf - acc)
        ece += w * gap; mce = max(mce, gap)
        rows.append({"bin": i, "conf": conf, "acc": acc, "count": int(m.sum())})
    return {"ece": float(ece), "mce": float(mce),
            "brier": float(brier_score_loss(y, p)), "bins": rows}


# ---------------------------------------------------------------------------
def net_benefit(y, p, thresholds=None):
    """Decision Curve Analysis: model net benefit vs 'treat all'/'none'."""
    y = np.asarray(y); p = np.asarray(p)
    n = len(y); prev = y.mean()
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    out = []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        w = pt / (1 - pt)
        nb_model = tp / n - fp / n * w
        nb_all = prev - (1 - prev) * w     # treat all
        out.append({"pt": float(pt), "nb_model": float(nb_model),
                    "nb_all": float(nb_all), "nb_none": 0.0})
    return out


# ---------------------------------------------------------------------------
def gate_safety(p, thr, patho_matrix, normal_mask=None):
    """Gate safety: among the images predicted NORMAL (p<thr), how many
    actually had pathology (miss). patho_matrix: array (N,) with #pathologies>0.

    Returns pathology miss-rate and filtered fraction.
    """
    p = np.asarray(p); has_patho = np.asarray(patho_matrix) > 0
    pred_normal = p < thr                 # gate says "normal" -> does not refer
    n = len(p)
    filtered = int(pred_normal.sum())
    missed = int((pred_normal & has_patho).sum())          # pathology missed!
    total_patho = int(has_patho.sum())
    return {
        "threshold": float(thr),
        "fraction_filtered": float(filtered / n),
        "n_filtered": filtered,
        "pathology_missed": missed,
        "miss_rate_of_pathology": float(missed / max(total_patho, 1)),
        "total_pathology": total_patho,
    }


# ---------------------------------------------------------------------------
def bootstrap_ci(y, p, fn, n_boot=2000, seed=42):
    """95% bootstrap CI for any function fn(y,p)->float."""
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try: vals.append(fn(y[idx], p[idx]))
        except Exception: pass
    if not vals:
        return {"mean": None, "lo": None, "hi": None}
    vals = np.array(vals)
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}
