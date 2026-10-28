"""Ablation study (Table 2): drop one module at a time.

    python scripts/run_ablation.py --set training.epochs=30

"w/o XAI Module" retrains the full network: the XAI module is post-hoc and
does not change the forward pass, so this row measures run-to-run variance.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from xfintech.config import deep_update, load_config, parse_cli_overrides
from xfintech.pipeline import get_dataframe, prepare_splits, run_experiment, update_results

VARIANTS = {
    "w/o Attention": {"model": {"use_attention": False}},
    "w/o Residual": {"model": {"use_residual": False}},
    "w/o Temporal LSTM": {"model": {"use_lstm": False}},
    "w/o Fairness Opt.": {"fairness": {"enabled": False}},
    "w/o XAI Module": {"data": {"seed": 7}},
    "Full Proposed": {},
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", action="append")
args = ap.parse_args()

base = load_config(args.config, parse_cli_overrides(args.set))
out = Path(base["output_dir"])
splits = prepare_splits(get_dataframe(base), base)

rows = []
for name, upd in VARIANTS.items():
    print(f"\n===== {name} =====")
    cfg = deep_update(base, upd)
    r = run_experiment(cfg, splits, verbose=0)
    m = r["metrics"]
    rows.append({"configuration": name, "accuracy": m["accuracy"], "f1": m["f1"], "auc_roc": m["auc_roc"],
                 "mcc": m["mcc"], "dp_difference": r["fairness"]["demographic_parity_difference"],
                 "eo_difference": r["fairness"]["equalized_odds_difference"]})
    print(json.dumps(rows[-1], indent=2))

update_results(out / "results.json", ablation=rows)
print(f"\nsaved ablation table to {out / 'results.json'}")
