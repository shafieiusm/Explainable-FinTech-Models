"""Run the XAI module on a trained model (Figures 4–6).

    python scripts/explain.py --run runs/default
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from xfintech import explain as X
from xfintech.model import build_model, predict_proba
from xfintech.pipeline import get_dataframe, prepare_splits, update_results

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/default")
ap.add_argument("--borrower", type=int, default=None, help="test-set row for the LIME explanation")
args = ap.parse_args()

run = Path(args.run)
cfg = json.loads((run / "config.json").read_text())
xcfg = cfg.get("xai", {})
splits = prepare_splits(get_dataframe(cfg), cfg)
pre = splits["pre"]
model = build_model(splits["X_train"].shape[1], cfg["model"])
model.load_weights(str(run / "model.weights.h5"))
f = lambda z: predict_proba(model, z)  # noqa: E731

rng = np.random.default_rng(0)
Xte, yte = splits["X_test"], splits["y_test"]
idx = rng.choice(len(Xte), size=min(xcfg.get("n_explain", 300), len(Xte)), replace=False)
Xe = Xte[idx]
bg = splits["X_train"][rng.choice(len(splits["X_train"]), xcfg.get("shap_background", 100), replace=False)]
groups, names = pre.feature_groups_, pre.feature_names_

print("[xai] SHAP (Eq. 43–44)")
phi = X.shap_values(f, bg, Xe, nsamples=xcfg.get("shap_samples", 300))
shap_imp = X.global_importance(phi, groups)

print("[xai] integrated gradients (Eq. 46)")
ig = X.integrated_gradients(model, Xe, steps=xcfg.get("ig_steps", 50))
ig_imp = X.global_importance(ig, groups)

print("[xai] attention explainability (Eq. 47)")
att_imp = X.attention_scores(model, Xe, groups)

print("[xai] permutation importance")
sub = rng.choice(len(Xte), size=min(5000, len(Xte)), replace=False)
perm_imp = X.permutation_importance(f, Xte[sub], yte[sub], groups)

ens = X.ensemble_importance(shap_imp, ig_imp, att_imp, perm_imp)

# attention scores per behavioural group (Figure 6, right panel)
att_groups = {}
for label, cols in xcfg.get("attention_groups", {}).items():
    present = [c for c in cols if c in att_imp.index]
    if present:
        att_groups[label] = float(att_imp[present].sum())

print("[xai] LIME (Eq. 45)")
p_test = f(Xte)
b = args.borrower if args.borrower is not None else int(np.argsort(np.abs(p_test - 0.5))[0])
lime = X.lime_explanation(f, splits["X_train"][:5000], Xte[b], names, xcfg.get("lime_num_features", 6))
# store with the paper's sign convention: positive = lowers default risk
lime_local = [[n, -w] for n, w in lime]

update_results(
    run / "results.json",
    feature_importance=ens.head(15).round(4).to_dict(),
    shap_importance=shap_imp.head(15).round(4).to_dict(),
    ig_importance=ig_imp.head(15).round(4).to_dict(),
    permutation_importance=perm_imp.head(15).round(4).to_dict(),
    attention_importance=att_groups or att_imp.head(6).round(4).to_dict(),
    lime_local=lime_local,
    lime_borrower={"test_row": b, "p_default": float(p_test[b])},
)
pd.DataFrame(phi, columns=names).to_csv(run / "shap_values.csv", index=False)
print("top features (ensemble):\n", ens.head(10).round(4).to_string())
