"""Five-fold stratified cross-validation on a 20 000-sample subset (Table 3).

    python scripts/run_cv.py --subset 20000 --folds 5
"""
import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from xfintech.config import load_config, parse_cli_overrides
from xfintech.data import stratified_kfold_indices
from xfintech.evaluate import classification_metrics, summarize_folds
from xfintech.model import build_model, predict_proba
from xfintech.pipeline import get_dataframe, set_seed, update_results
from xfintech.preprocessing import TabularPreprocessor
from xfintech.train import Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", action="append")
ap.add_argument("--subset", type=int, default=20_000)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--threshold", type=float, default=0.5)
args = ap.parse_args()

cfg = load_config(args.config, parse_cli_overrides(args.set))
d = cfg["data"]
df = get_dataframe(cfg)
df = df.drop(columns=[c for c in d.get("id_columns", []) if c in df])
df = df.groupby(d["target"], group_keys=False).sample(frac=min(1.0, args.subset / len(df)), random_state=d["seed"])
df = df.reset_index(drop=True)
y_all = df[d["target"]].to_numpy()

folds = []
for k, (tr_idx, te_idx) in enumerate(stratified_kfold_indices(y_all, args.folds, d["seed"]), start=1):
    set_seed(d["seed"] + k)
    tr, te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    n_val = int(0.15 * len(tr))
    va, tr = tr.iloc[:n_val], tr.iloc[n_val:]
    pre = TabularPreprocessor(d["target"], d.get("protected_attribute"), d.get("privileged_value"),
                              [c for c in d.get("drop_columns", []) if c in df]).fit(tr)
    model = build_model(pre.transform(tr).shape[1], cfg["model"])
    Trainer(model, cfg["training"], cfg.get("fairness"), verbose=0).fit(
        pre.transform(tr), pre.labels(tr), pre.protected(tr), pre.transform(va), pre.labels(va))
    m = classification_metrics(pre.labels(te), predict_proba(model, pre.transform(te)), args.threshold)
    m["fold"] = k
    folds.append(m)
    print(f"fold {k}: acc {m['accuracy']:.2f} | P {m['precision']:.2f} | R {m['recall']:.2f} | "
          f"F1 {m['f1']:.2f} | AUC {m['auc_roc']:.2f}")

summary = summarize_folds(folds)
print({k: f"{v['mean']:.2f} ± {v['std']:.2f}" for k, v in summary.items()})
update_results(Path(cfg["output_dir"]) / "results.json", cross_validation={"folds": folds, "summary": summary})
