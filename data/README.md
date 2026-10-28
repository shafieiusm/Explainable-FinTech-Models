# Data

## FinTech Credit Risk Dataset (paper, ref. [33])

* ~300 000 borrower records, 51 demographic, financial, behavioural and alternative-data attributes
* target `default` (≈ 86.29 % non-default / 13.71 % default)
* link: https://docs.google.com/spreadsheets/d/1mx1Oin1HUaaqQx8_rSoCK6oJbJsenZs1/edit?usp=sharing

Download it with `python scripts/download_data.py`, or open the link, choose
**File → Download → CSV**, and save it as `data/fintech_credit_risk.csv`.
Then check the column names in `configs/default.yaml`:

| config key | meaning |
|---|---|
| `data.target` | default label column (1 = default) |
| `data.protected_attribute` / `privileged_value` | group attribute used by the fairness module |
| `data.id_columns` | identifiers to drop |
| `data.drop_columns` | post-outcome columns that leak the label (e.g. loan approval status) |
| `xai.attention_groups` | which columns make up each behavioural group in Figure 6 |

Numeric vs. categorical columns are detected automatically from their dtype.

## Synthetic data

`sample_synthetic.csv` (2 000 rows) is bundled for smoke tests. Create a
full-size synthetic dataset with the same schema:

```bash
python scripts/generate_synthetic_data.py --n 300000 --out data/fintech_credit_risk.csv
```

If `data.path` does not exist, `scripts/train.py` generates synthetic data
automatically. Results on synthetic data are **not** the paper's results.
