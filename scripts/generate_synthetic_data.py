"""Create a synthetic borrower dataset with the paper's schema.

    python scripts/generate_synthetic_data.py --n 300000 --out data/fintech_credit_risk.csv
"""
import argparse

import _bootstrap  # noqa: F401
from xfintech.data import make_synthetic_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300_000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="data/fintech_credit_risk.csv")
args = ap.parse_args()

df = make_synthetic_dataset(n=args.n, seed=args.seed)
df.to_csv(args.out, index=False)
n_attr = df.shape[1] - 2  # minus id and target
print(f"wrote {args.out}: {len(df):,} rows, {n_attr} attributes, default rate {df['default'].mean():.2%}")
