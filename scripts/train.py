"""Train and evaluate the full Explainable FinTech model.

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --set training.epochs=5 --set data.path=data/sample.csv
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from xfintech.config import load_config, parse_cli_overrides
from xfintech.pipeline import get_dataframe, prepare_splits, run_experiment, update_results
from xfintech.plots import fig_training_curves

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", action="append", help="override, e.g. --set model.dropout=0.4")
ap.add_argument("--n-synthetic", type=int, default=300_000, help="rows to synthesise if data file is missing")
args = ap.parse_args()

cfg = load_config(args.config, parse_cli_overrides(args.set))
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)

df = get_dataframe(cfg, n_synthetic=args.n_synthetic)
splits = prepare_splits(df, cfg)
print(f"[data] train {len(splits['X_train']):,} | val {len(splits['X_val']):,} | test {len(splits['X_test']):,} "
      f"| {splits['X_train'].shape[1]} encoded features | default rate {splits['y_train'].mean():.2%}")

res = run_experiment(cfg, splits)
res["model"].save_weights(str(out / "model.weights.h5"))
splits["pre"].save(out / "preprocessor.json")
(out / "config.json").write_text(json.dumps(cfg, indent=2))
(out / "history.json").write_text(json.dumps(res["history"], indent=2))
fig_training_curves(res["history"], out / "training_curves.png")

update_results(out / "results.json", test_metrics=res["metrics"], fairness=res["fairness"])
print("\n[test]", json.dumps(res["metrics"], indent=2))
print("[fairness]", json.dumps(res["fairness"], indent=2))
