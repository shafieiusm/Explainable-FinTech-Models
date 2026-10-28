"""Draw every figure.

    python scripts/make_figures.py                         # paper's reported values -> figures/
    python scripts/make_figures.py --results runs/default/results.json --out runs/default/figures
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from xfintech import plots as P
from xfintech.data import load_dataset, make_synthetic_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/paper_results.json")
ap.add_argument("--data", default="data/fintech_credit_risk.csv", help="dataset for the credit-score histogram")
ap.add_argument("--out", default="figures")
args = ap.parse_args()

R = json.loads(Path(args.results).read_text())
out = Path(args.out)
made = []

if Path(args.data).exists():
    df, note = load_dataset(args.data), None
else:
    df = make_synthetic_dataset(n=100_000, seed=42)
    note = "Drawn from the synthetic schema-compatible dataset (data file not found)."
made.append(P.fig_credit_score_distribution(df["credit_score"], out / "fig3_credit_score_distribution.png", note=note))

if "feature_importance" in R:
    made.append(P.fig_feature_importance(R["feature_importance"], out / "fig4_feature_importance.png"))
if "shap_importance" in R:
    made.append(P.fig_shap_importance(R["shap_importance"], out / "fig5_shap_importance.png"))
if "lime_local" in R and "attention_importance" in R:
    made.append(P.fig_lime_attention(R["lime_local"], R["attention_importance"], out / "fig6_lime_attention.png"))
if "lending" in R:
    made.append(P.fig_lending(R["lending"], out / "fig7_personalized_lending.png"))
if "dynamic_risk" in R:
    made.append(P.fig_dynamic_risk(R["dynamic_risk"], out / "fig8_dynamic_risk.png"))
if "ablation" in R:
    made.append(P.fig_ablation(R["ablation"], out / "table2_ablation.png"))
if "cross_validation" in R:
    made.append(P.fig_cross_validation(R["cross_validation"]["folds"], out / "table3_cross_validation.png"))

for m in made:
    print("wrote", m)
