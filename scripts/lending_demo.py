"""Personalised lending recommendations and dynamic risk adaptation (Figures 7–8).

    python scripts/lending_demo.py --run runs/default
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from xfintech.lending import LendingEngine, dynamic_risk_update
from xfintech.model import build_model, predict_proba
from xfintech.pipeline import get_dataframe, prepare_splits, update_results
from xfintech.uncertainty import mc_dropout_predict

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/default")
ap.add_argument("--n", type=int, default=10)
args = ap.parse_args()

run = Path(args.run)
cfg = json.loads((run / "config.json").read_text())
splits = prepare_splits(get_dataframe(cfg), cfg)
pre, te = splits["pre"], splits["frames"]["test"]
model = build_model(splits["X_train"].shape[1], cfg["model"])
model.load_weights(str(run / "model.weights.h5"))
engine = LendingEngine.from_config(cfg["lending"])

# ---- Figure 7: ten borrowers spread over the risk range ---------------------
p = predict_proba(model, splits["X_test"])
rng = np.random.default_rng(1)
qs = np.linspace(0.05, 0.995, args.n)
pick = [int(np.argsort(np.abs(p - np.quantile(p, q)))[0]) for q in qs]
pick = list(rng.permutation(pick))
mc = mc_dropout_predict(model, splits["X_test"][pick], T=30)          # Eq. 40
ids = [f"B{i + 1:03d}" for i in range(len(pick))]
income = te["monthly_income"].to_numpy()[pick] if "monthly_income" in te else None
spend = te["monthly_spending"].to_numpy()[pick] if "monthly_spending" in te else None
table = engine.recommend(ids, mc["mean"], income, spend)
table["pd_std"] = mc["std"].round(4)
print(table.to_string(index=False))
table.to_csv(run / "lending_recommendations.csv", index=False)

# ---- Figure 8: one borrower over five months ---------------------------------
row = te.iloc[[pick[int(np.argsort(mc["mean"])[len(pick) // 2])]]].copy()
months = []
scen = [
    {},                                                                    # month 1: as observed
    {"DTI": +0.06, "credit_utilisation_ratio": +0.10},                     # month 2: more leverage
    {"num_late_payments": +1, "num_missed_payments": +1},                  # month 3: missed payment
    {"num_missed_payments": +1, "credit_utilisation_ratio": +0.15},        # month 4: distress
    {"DTI": -0.20, "credit_utilisation_ratio": -0.35, "num_missed_payments": -2, "num_late_payments": -1},  # month 5: recovery
]
cur = row.copy()
for change in scen:
    for col, delta in change.items():
        if col in cur:
            cur[col] = np.clip(cur[col].astype(float) + delta, 0, None)
    months.append(cur.copy())
f_t = [float(predict_proba(model, pre.transform(m))[0]) for m in months]
lam = cfg["lending"].get("lambda_dynamic", 0.3)
initial, updated, prev = [], [], f_t[0]
for r in f_t:
    initial.append(round(prev, 4))
    prev = dynamic_risk_update(prev, r, lam)                              # Eq. 42
    updated.append(round(prev, 4))
dyn = {"labels": [f"Month {i + 1}" for i in range(len(f_t))], "initial": initial, "updated": updated,
       "model_score": [round(v, 4) for v in f_t]}
print("dynamic risk:", dyn)

update_results(run / "results.json",
               lending=table[["borrower_id", "credit_risk_score", "interest_rate_pct", "risk_band"]].to_dict("records"),
               dynamic_risk=dyn)
