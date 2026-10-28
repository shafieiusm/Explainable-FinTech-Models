"""Figure generation (Figures 3–8 of the paper plus ablation / CV charts).

Every function takes plain Python data (the schema of
``results/paper_results.json``), so the same code draws the paper's reported
numbers or the numbers produced by your own run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# palette — categorical slots, one sequential hue, reserved status colours
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SEQ = ["#0d366b", "#184f95", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]
GOOD, WARN, CRIT = "#1f8a4c", "#d69a00", "#c8382f"
INK, INK2, GRID = "#1b1b1b", "#55534e", "#e4e2dc"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, out: str | Path):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _pretty(name: str) -> str:
    return name if any(ch.isupper() for ch in name) else name.replace("_", " ").capitalize()


def _hbar(ax, labels, values, fmt="{:.4f}", color=None, signed=False):
    labels, values = list(labels)[::-1], np.asarray(list(values)[::-1], dtype=float)
    if signed:
        colors = [GOOD if v >= 0 else CRIT for v in values]
    else:  # one sequential hue: darkest = most important
        ranks = np.argsort(np.argsort(-values))
        colors = [color or SEQ[min(r, len(SEQ) - 1)] for r in ranks]
    ax.barh(labels, values, color=colors, height=0.68, edgecolor="white", linewidth=1.5)
    span = (np.abs(values).max() or 1.0)
    for i, v in enumerate(values):
        ha = "left" if v >= 0 else "right"
        ax.text(v + (0.012 if v >= 0 else -0.012) * span, i, fmt.format(v), va="center", ha=ha,
                fontsize=9.5, color=INK)
    ax.grid(axis="y", visible=False)
    if signed:
        ax.axvline(0, color=INK2, lw=1)
        lim = span * 1.25
        ax.set_xlim(-lim, lim)
    else:
        ax.set_xlim(0, span * 1.18)


# ----------------------------------------------------------------- Figure 3
def fig_credit_score_distribution(scores, out, title="Distribution of borrower credit scores", note=None):
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(scores, bins=30, color=BLUE, edgecolor="white", linewidth=1.2)
    ax.axvline(np.median(scores), color=INK, lw=1.5, ls="--")
    ax.text(np.median(scores) + 5, ax.get_ylim()[1] * 0.93, f"median {np.median(scores):.0f}", color=INK, fontsize=9.5)
    ax.set(title=title, xlabel="Credit score", ylabel="Number of borrowers")
    ax.grid(axis="x", visible=False)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8.5, color=INK2)
    return _save(fig, out)


# ----------------------------------------------------------------- Figure 4
def fig_feature_importance(importance: dict, out, top=10, title="Top feature importance for credit-risk prediction"):
    items = sorted(importance.items(), key=lambda kv: -kv[1])[:top]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _hbar(ax, [k for k, _ in items], [v for _, v in items])
    ax.set(title=title, xlabel="Ensemble importance score")
    return _save(fig, out)


# ----------------------------------------------------------------- Figure 5
def fig_shap_importance(importance: dict, out, top=10, title="SHAP-based feature importance"):
    items = sorted(importance.items(), key=lambda kv: -kv[1])[:top]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _hbar(ax, [_pretty(k) for k, _ in items], [v for _, v in items], fmt="{:.3f}")
    ax.set(title=title, xlabel="Mean |SHAP value|  (Eq. 44)")
    return _save(fig, out)


# ----------------------------------------------------------------- Figure 6
def fig_lime_attention(lime_local, attention: dict, out):
    """Left: LIME weights for one borrower — convention: positive (green) = supports
    repayment / lowers default risk, negative (red) = raises risk.
    Right: attention explainability scores (Eq. 47)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    _hbar(a1, [_pretty(n) for n, _ in lime_local], [v for _, v in lime_local], fmt="{:+.2f}", signed=True)
    a1.set(title="LIME local explanation (one borrower)", xlabel="← raises default risk   |   lowers default risk →")
    items = sorted(attention.items(), key=lambda kv: -kv[1])
    _hbar(a2, [_pretty(k) for k, _ in items], [v for _, v in items], fmt="{:.2f}")
    a2.set(title="Attention explainability score", xlabel="Attention importance (Eq. 47)")
    return _save(fig, out)


# ----------------------------------------------------------------- Figure 7
def fig_lending(rows, out, title="Personalised lending recommendations"):
    """Two aligned panels (one y-scale each): credit-risk score by band, and the
    adaptive interest rate from Eq. 55."""
    ids = [r["borrower_id"] for r in rows]
    crs = np.array([r["credit_risk_score"] for r in rows])
    rate = np.array([r["interest_rate_pct"] for r in rows])
    band = [r["risk_band"] for r in rows]
    cmap = {"Low Risk": GOOD, "Medium Risk": WARN, "High Risk": CRIT}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.bar(ids, crs, color=[cmap[b] for b in band], width=0.7, edgecolor="white", linewidth=1.5)
    for i, v in enumerate(crs):
        a1.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    a1.legend(handles, cmap.keys(), loc="upper left", ncol=3)
    a1.set(title=title, ylabel="Credit risk score", ylim=(0, max(crs) * 1.2))
    a1.grid(axis="x", visible=False)
    a2.plot(ids, rate, color=BLUE, lw=2, marker="o", ms=7, markeredgecolor="white", markeredgewidth=1.5)
    for i, v in enumerate(rate):
        a2.text(i, v + 1.3, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    a2.set(xlabel="Borrower ID", ylabel="Interest rate (%)", ylim=(0, max(rate) * 1.25))
    return _save(fig, out)


# ----------------------------------------------------------------- Figure 8
def fig_dynamic_risk(dyn: dict, out, title="Dynamic credit-risk adaptation"):
    labels, init, upd = dyn["labels"], np.array(dyn["initial"]), np.array(dyn["updated"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for lo, hi, c, name in [(0, 0.3, GOOD, "low"), (0.3, 0.6, WARN, "medium"), (0.6, 1, CRIT, "high")]:
        ax.axhspan(lo, hi, color=c, alpha=0.06, lw=0)
        ax.text(len(labels) - 0.25, min(hi, 0.8) - 0.012, f"{name} risk", fontsize=8.5, color=c, ha="right", va="top")
    ax.set_xlim(-0.3, len(labels) - 0.2)
    ax.plot(x, init, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor="white", markeredgewidth=1.5,
            label="Initial risk score")
    ax.plot(x, upd, color=ORANGE, lw=2, marker="s", ms=8, markeredgecolor="white", markeredgewidth=1.5,
            label="Updated risk score (Eq. 42)")
    for xi, v in zip(x, upd):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set(title=title, xlabel="Time interval", ylabel="Risk score", ylim=(0.2, 0.8))
    ax.legend(loc="upper left")
    return _save(fig, out)


# ------------------------------------------------------------ Table 2 chart
def fig_ablation(rows, out, title="Ablation study (Table 2)"):
    names = [r["configuration"] for r in rows]
    metrics = [("accuracy", "Accuracy (%)"), ("f1", "F1-score (%)"), ("auc_roc", "AUC-ROC (%)"), ("mcc", "MCC")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, (k, lab) in zip(axes, metrics):
        vals = np.array([r[k] for r in rows], dtype=float)
        colors = [BLUE if "Full" in n else "#86b6ef" for n in names]
        # dot plot: the axis does not start at zero, so bars would exaggerate the gaps
        ax.scatter(vals, names, s=[110 if "Full" in n else 70 for n in names], c=colors, zorder=3,
                   edgecolors="white", linewidths=1.5)
        lo = vals.min() * (0.9 if k != "mcc" else 0.85)
        ax.set_xlim(lo, vals.max() + (vals.max() - lo) * 0.25)
        for i, v in enumerate(vals):
            ax.text(v + (vals.max() - lo) * 0.05, i, f"{v:.2f}" if k != "mcc" else f"{v:.3f}", va="center", fontsize=9)
        ax.set_title(lab, fontsize=11)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontweight="bold")
    return _save(fig, out)


# ------------------------------------------------------------ Table 3 chart
def fig_cross_validation(folds, out, title="Five-fold stratified cross-validation (Table 3)"):
    keys = [("accuracy", "Accuracy", BLUE), ("precision", "Precision", AQUA), ("recall", "Recall", ORANGE),
            ("f1", "F1-score", VIOLET), ("auc_roc", "AUC-ROC", "#eda100")]
    x = np.arange(len(folds))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for j, (k, lab, c) in enumerate(keys):
        ax.bar(x + (j - 2) * w, [f[k] for f in folds], width=w - 0.02, color=c, label=lab, edgecolor="white")
    ax.set_xticks(x, [f"Fold {f.get('fold', i + 1)}" for i, f in enumerate(folds)])
    ax.set(title=title, ylabel="Score (%)", ylim=(0, 110))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    return _save(fig, out)


def fig_training_curves(history, out, title="Training history"):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], color=BLUE, lw=2, label="train BCE")
    a1.plot(ep, [h["val_loss"] for h in history], color=ORANGE, lw=2, label="val BCE")
    a1.set(title="Loss", xlabel="Epoch")
    a1.legend()
    a2.plot(ep, [h["val_auc"] for h in history], color=AQUA, lw=2)
    a2.set(title="Validation AUC-ROC", xlabel="Epoch")
    fig.suptitle(title, fontweight="bold")
    return _save(fig, out)
