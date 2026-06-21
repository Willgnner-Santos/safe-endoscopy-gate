# -*- coding: utf-8 -*-
"""Bilingual figures (EN/PT), one per file, legend without overlap.

Each function plots and saves TWO versions (en and pt) in figures/en and figures/pt.
Anti-overlap rules:
  - constrained_layout=True
  - legend outside the axis when there are many series (bbox_to_anchor)
  - savefig(bbox_inches='tight')
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import config as C

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": C.DPI,
                     "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# Texts by language -----------------------------------------------------------
TXT = {
    "roc": {
        "en": dict(title="ROC curve — Normal vs. Altered gate", x="False Positive Rate",
                   y="True Positive Rate (Sensitivity)", auc="AUROC"),
        "pt": dict(title="Curva ROC — gate Normal vs. Alterado", x="Taxa de Falsos Positivos",
                   y="Taxa de Verdadeiros Positivos (Sensibilidade)", auc="AUROC"),
    },
    "pr": {
        "en": dict(title="Precision–Recall curve", x="Recall (Sensitivity)", y="Precision", auc="AUPRC"),
        "pt": dict(title="Curva Precisão–Revocação", x="Revocação (Sensibilidade)", y="Precisão", auc="AUPRC"),
    },
    "calib": {
        "en": dict(title="Reliability diagram (calibration)", x="Predicted probability",
                   y="Observed frequency", perfect="Perfect calibration"),
        "pt": dict(title="Diagrama de confiabilidade (calibração)", x="Probabilidade prevista",
                   y="Frequência observada", perfect="Calibração perfeita"),
    },
    "nb": {
        "en": dict(title="Decision curve (net benefit)", x="Threshold probability",
                   y="Net benefit", model="Model (gate)", all="Treat all", none="Treat none"),
        "pt": dict(title="Curva de decisão (benefício líquido)", x="Probabilidade de limiar",
                   y="Benefício líquido", model="Modelo (gate)", all="Encaminhar todos", none="Encaminhar nenhum"),
    },
    "safety": {
        "en": dict(title="Gate safety: pathology missed vs. workload filtered",
                   x="Fraction of exams filtered as Normal", y="Pathology miss rate"),
        "pt": dict(title="Segurança do gate: patologia perdida vs. carga filtrada",
                   x="Fração de exames filtrados como Normal", y="Taxa de patologia perdida"),
    },
    "modelcmp": {
        "en": dict(title="Model comparison (mean ± SD over 5 folds)", y="Score", x="Model"),
        "pt": dict(title="Comparação de modelos (média ± DP em 5 folds)", y="Métrica", x="Modelo"),
    },
    "train": {
        "en": dict(title="Training curves", x="Epoch", loss="Loss", metric="Val AUPRC"),
        "pt": dict(title="Curvas de treino", x="Época", loss="Perda", metric="AUPRC val"),
    },
}


def _save(fig, name, lang):
    out = (C.FIG_EN if lang == "en" else C.FIG_PT) / f"{name}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _both(plot_fn, name):
    outs = []
    for lang in C.LANGS:
        fig = plot_fn(lang); outs.append(_save(fig, name, lang))
    return outs


# --- ROC -----------------------------------------------------------------
def plot_roc(curves, name="roc"):
    """curves: list of dict(model, fpr, tpr, auc)."""
    def mk(lang):
        t = TXT["roc"][lang]
        fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
        for c in curves:
            ax.plot(c["fpr"], c["tpr"], lw=1.8, label=f"{c['model']} ({t['auc']}={c['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set(title=t["title"], xlabel=t["x"], ylabel=t["y"], xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        return fig
    return _both(mk, name)


def plot_pr(curves, name="pr"):
    def mk(lang):
        t = TXT["pr"][lang]
        fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
        for c in curves:
            ax.plot(c["recall"], c["precision"], lw=1.8, label=f"{c['model']} ({t['auc']}={c['auc']:.3f})")
        ax.set(title=t["title"], xlabel=t["x"], ylabel=t["y"], xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        return fig
    return _both(mk, name)


def plot_calibration(bins, ece, name="calibration"):
    def mk(lang):
        t = TXT["calib"][lang]
        fig, ax = plt.subplots(figsize=(5.0, 4.8), constrained_layout=True)
        conf = [b["conf"] for b in bins if b["conf"] is not None]
        acc  = [b["acc"]  for b in bins if b["acc"] is not None]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label=t["perfect"])
        ax.plot(conf, acc, "o-", lw=1.8, label=f"ECE={ece:.3f}")
        ax.set(title=t["title"], xlabel=t["x"], ylabel=t["y"], xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=9)
        return fig
    return _both(mk, name)


def plot_net_benefit(nb, name="decision_curve"):
    def mk(lang):
        t = TXT["nb"][lang]
        pt = [d["pt"] for d in nb]
        fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
        ax.plot(pt, [d["nb_model"] for d in nb], lw=2, label=t["model"])
        ax.plot(pt, [d["nb_all"] for d in nb], "--", lw=1.2, label=t["all"])
        ax.axhline(0, color="k", lw=1, label=t["none"])
        ax.set(title=t["title"], xlabel=t["x"], ylabel=t["y"], xlim=(0, 1))
        ymin = min(0, min(d["nb_model"] for d in nb)) - 0.02
        ax.set_ylim(ymin, max(d["nb_model"] for d in nb) + 0.05)
        ax.legend(loc="upper right", fontsize=9)
        return fig
    return _both(mk, name)


def plot_gate_safety(frac_filtered, miss_rate, name="gate_safety"):
    def mk(lang):
        t = TXT["safety"][lang]
        fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
        ax.plot(frac_filtered, miss_rate, "o-", lw=1.8)
        ax.set(title=t["title"], xlabel=t["x"], ylabel=t["y"])
        return fig
    return _both(mk, name)


def plot_model_comparison(model_names, means, sds, metric_label, name="model_comparison"):
    def mk(lang):
        t = TXT["modelcmp"][lang]
        fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
        x = np.arange(len(model_names))
        ax.bar(x, means, yerr=sds, capsize=4, color="#4C78A8", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
        ax.set(title=t["title"], ylabel=f"{t['y']} ({metric_label})", xlabel=t["x"])
        for xi, m in zip(x, means):
            ax.text(xi, m + (max(sds) if len(sds) else 0.005) + 0.005, f"{m:.3f}",
                    ha="center", fontsize=8)
        return fig
    return _both(mk, name)


def plot_training(history, model_key, fold, name=None):
    """history: dict(epoch[], train_loss[], val_loss[], val_auprc[]). Saves to figures/training."""
    name = name or f"train_{model_key}_fold{fold}"
    for lang in C.LANGS:
        t = TXT["train"][lang]
        fig, ax1 = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
        ep = history["epoch"]
        l1, = ax1.plot(ep, history["train_loss"], "C0-", label=f"{t['loss']} (train)")
        l2, = ax1.plot(ep, history["val_loss"], "C0--", label=f"{t['loss']} (val)")
        ax1.set_xlabel(t["x"]); ax1.set_ylabel(t["loss"])
        ax2 = ax1.twinx()
        l3, = ax2.plot(ep, history["val_auprc"], "C1-", label=t["metric"])
        ax2.set_ylabel(t["metric"]); ax2.grid(False)
        ax1.set_title(f"{t['title']} — {model_key} (fold {fold})")
        ax1.legend(handles=[l1, l2, l3], loc="center right", fontsize=8, framealpha=0.9)
        fig.savefig(C.FIG_TRAIN / f"{name}_{lang}.png", bbox_inches="tight")
        plt.close(fig)
