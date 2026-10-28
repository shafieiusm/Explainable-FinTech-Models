"""Dataset loading, a schema-compatible synthetic generator, and stratified splits.

The paper uses the *FinTech Credit Risk Dataset* (~300 000 borrowers, 51
attributes, ~13.71 % defaults).  Export it from the link in ``data/README.md``
and point ``data.path`` in the config at the file.  When the file is not
available, :func:`make_synthetic_dataset` produces data with the same kind of
columns so that the whole pipeline can be run end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RATE = 0.1371  # Section 4.2


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset (scripts/download_data.py) "
            "or create synthetic data with scripts/generate_synthetic_data.py."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def stratified_split(
    df: pd.DataFrame, target: str, train: float = 0.70, val: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70 / 15 / 15 stratified split (Section 4.1)."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, grp in df.groupby(target):
        idx = rng.permutation(grp.index.to_numpy())
        n_tr = int(round(train * len(idx)))
        n_va = int(round(val * len(idx)))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr : n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va :])
    out = []
    for k in ("train", "val", "test"):
        ids = rng.permutation(np.concatenate(parts[k]))
        out.append(df.loc[ids].reset_index(drop=True))
    return tuple(out)  # type: ignore[return-value]


def stratified_kfold_indices(y: np.ndarray, k: int = 5, seed: int = 42):
    """Yield ``(train_idx, test_idx)`` for stratified k-fold CV (Section 4.5)."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for f, chunk in enumerate(np.array_split(idx, k)):
            folds[f].append(chunk)
    folds = [np.concatenate(f) for f in folds]
    for f in range(k):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != f])
        yield rng.permutation(train_idx), test_idx


# --------------------------------------------------------------------------- #
# Synthetic data with the same schema as the paper's dataset
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(n: int = 300_000, seed: int = 42, missing_rate: float = 0.02) -> pd.DataFrame:
    """Generate borrower records with demographic, financial, behavioural and
    alternative-data attributes (51 attributes + id + target).

    Default risk is driven mainly by credit score, DTI and missed payments, so
    the explainability modules have a known ground truth to recover.
    """
    rng = np.random.default_rng(seed)
    U = rng.uniform
    N = rng.normal

    def choice(opts, p):
        return rng.choice(opts, size=n, p=p)

    # ---- demographic --------------------------------------------------------
    age = np.clip(N(40, 11, n), 21, 75).round()
    gender = choice(["Male", "Female"], [0.55, 0.45])
    marital = choice(["Single", "Married", "Divorced", "Widowed"], [0.38, 0.48, 0.11, 0.03])
    education = choice(["High School", "Bachelor", "Master", "PhD", "Other"], [0.30, 0.42, 0.18, 0.04, 0.06])
    employment = choice(["Salaried", "Self-Employed", "Freelancer", "Unemployed", "Retired"],
                        [0.56, 0.20, 0.12, 0.06, 0.06])
    dependents = rng.poisson(1.2, n).clip(0, 6)
    region = choice(["Urban", "Semi-Urban", "Rural"], [0.52, 0.30, 0.18])
    housing = choice(["Own", "Rent", "Mortgage", "Family"], [0.28, 0.36, 0.26, 0.10])

    # ---- income & employment -----------------------------------------------
    emp_factor = pd.Series(employment).map(
        {"Salaried": 1.0, "Self-Employed": 1.15, "Freelancer": 0.85, "Unemployed": 0.35, "Retired": 0.6}
    ).to_numpy()
    edu_factor = pd.Series(education).map(
        {"High School": 0.8, "Bachelor": 1.0, "Master": 1.25, "PhD": 1.45, "Other": 0.85}
    ).to_numpy()
    gender_gap = np.where(gender == "Male", 1.06, 0.94)  # mild historical pay gap -> fairness signal
    annual_income = np.exp(N(10.9, 0.45, n)) * emp_factor * edu_factor * gender_gap
    monthly_income = annual_income / 12.0
    years_employed = np.clip(N(8, 6, n), 0, 45).round(1)

    # ---- credit bureau -------------------------------------------------------
    latent = N(0, 1, n)  # latent creditworthiness
    credit_score = np.clip(655 + 80 * latent + N(0, 25, n), 300, 850).round()
    credit_history_length = np.clip(N(110, 60, n) + 20 * latent, 6, 480).round()
    credit_limit = np.clip(annual_income * U(0.15, 0.6, n), 500, None).round(-1)
    util = np.clip(0.35 - 0.12 * latent + N(0, 0.15, n), 0.0, 1.0)
    credit_used = (credit_limit * util).round(-1)
    credit_utilisation_ratio = credit_used / credit_limit  # Eq. 54
    num_hard_inquiries = rng.poisson(np.clip(1.5 - 0.5 * latent, 0.1, None))
    num_late_payments = rng.poisson(np.clip(1.2 - 0.8 * latent, 0.05, None))
    num_missed_payments = rng.poisson(np.clip(0.6 - 0.7 * latent, 0.02, None))
    bankruptcy_history = (U(0, 1, n) < 1 / (1 + np.exp(3.8 + 1.3 * latent))).astype(int)
    months_since_last_default = np.where(
        (num_missed_payments > 0) | (bankruptcy_history == 1), np.clip(N(30, 20, n), 1, 120).round(), 999
    )
    repayment_history_score = np.clip(0.8 + 0.1 * latent - 0.05 * num_missed_payments + N(0, 0.05, n), 0, 1)
    repayment_trend = np.clip(N(0, 1, n) * 0.5 + 0.4 * latent, -3, 3)

    # ---- debt & loan ---------------------------------------------------------
    existing_loans = rng.poisson(1.3, n)
    monthly_debt = np.clip(monthly_income * np.clip(N(0.30, 0.12, n) - 0.04 * latent, 0.02, 0.95), 0, None)
    DTI = monthly_debt / monthly_income  # Eq. 53
    total_outstanding_debt = monthly_debt * U(12, 60, n)
    mortgage_flag = (housing == "Mortgage").astype(int)
    loan_amount = np.clip(annual_income * U(0.1, 1.2, n), 1_000, None).round(-2)
    loan_term_months = choice([12, 24, 36, 48, 60], [0.12, 0.22, 0.34, 0.16, 0.16])
    loan_purpose = choice(["Debt Consolidation", "Home", "Car", "Education", "Business", "Medical", "Personal"],
                          [0.28, 0.14, 0.14, 0.10, 0.12, 0.07, 0.15])
    interest_rate_requested = np.clip(N(11, 3.5, n), 3, 30).round(2)
    collateral_flag = (U(0, 1, n) < 0.35).astype(int)
    loan_to_income = loan_amount / annual_income

    # ---- behavioural ---------------------------------------------------------
    monthly_spending = monthly_income * np.clip(N(0.62, 0.15, n) - 0.04 * latent, 0.1, 1.5)
    spending_behaviour_index = np.clip(monthly_spending / monthly_income, 0, 2)
    savings_ratio = np.clip(1 - spending_behaviour_index - DTI + N(0, 0.05, n), -0.5, 0.8)
    transaction_volume = rng.poisson(np.clip(60 + 15 * latent, 5, None))
    transaction_stability = np.clip(0.65 + 0.12 * latent + N(0, 0.1, n), 0, 1)
    avg_account_balance = np.clip(monthly_income * (0.5 + savings_ratio) * U(0.5, 2.0, n), 0, None)
    overdraft_count = rng.poisson(np.clip(0.8 - 0.6 * latent, 0.02, None))
    loan_repayment_delay_days = np.clip(rng.exponential(np.clip(4 - 3 * latent, 0.3, None)), 0, 120).round()

    # ---- alternative data ----------------------------------------------------
    mobile_money_usage = np.clip(N(0.5, 0.2, n) + np.where(region == "Rural", 0.15, 0), 0, 1)
    mobile_payment_frequency = rng.poisson(np.clip(25 + 6 * latent, 1, None))
    digital_wallet_balance = np.clip(rng.exponential(300, n) * (1 + 0.2 * latent), 0, None)
    utility_bill_consistency = np.clip(0.82 + 0.08 * latent + N(0, 0.07, n), 0, 1)
    rent_payment_consistency = np.clip(0.8 + 0.08 * latent + N(0, 0.08, n), 0, 1)
    ecommerce_txn_count = rng.poisson(12, n)
    social_financial_score = np.clip(N(0.5, 0.15, n) + 0.05 * latent, 0, 1)
    account_category = choice(["Basic", "Standard", "Premium"], [0.45, 0.40, 0.15])

    # ---- default label (Eq. 3) --------------------------------------------------
    z = (
        -3.2 * (credit_score - 655) / 80
        + 2.1 * (DTI - 0.30) / 0.12
        + 1.1 * num_missed_payments
        + 0.9 * bankruptcy_history
        + 0.35 * num_late_payments
        + 0.9 * (credit_utilisation_ratio - 0.35) / 0.15
        - 0.5 * (repayment_history_score - 0.8) / 0.08
        - 0.3 * (np.log(monthly_income) - 8.4)
        + 0.25 * (employment == "Unemployed")
        + 0.4 * (months_since_last_default < 24)
        - 0.2 * (utility_bill_consistency - 0.82) / 0.1
        + N(0, 0.6, n)
    )
    # calibrate intercept to the paper's default rate via bisection
    lo, hi = -20.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(z + mid)))).mean()
        lo, hi = (mid, hi) if rate < DEFAULT_RATE else (lo, mid)
    p_default = 1 / (1 + np.exp(-(z + mid)))
    default = (U(0, 1, n) < p_default).astype(int)
    loan_approval_status = np.where(default == 1, "Rejected", "Approved")  # leaks label -> dropped

    df = pd.DataFrame({
        "borrower_id": [f"B{i:06d}" for i in range(1, n + 1)],
        "age": age, "gender": gender, "marital_status": marital, "education_level": education,
        "employment_type": employment, "dependents": dependents, "region": region, "housing_status": housing,
        "annual_income": annual_income.round(2), "monthly_income": monthly_income.round(2),
        "years_employed": years_employed,
        "credit_score": credit_score, "credit_history_length_months": credit_history_length,
        "credit_limit": credit_limit, "credit_used": credit_used,
        "credit_utilisation_ratio": credit_utilisation_ratio.round(4), "num_hard_inquiries": num_hard_inquiries,
        "num_late_payments": num_late_payments, "num_missed_payments": num_missed_payments,
        "bankruptcy_history": bankruptcy_history, "months_since_last_default": months_since_last_default,
        "repayment_history_score": repayment_history_score.round(4), "repayment_trend": repayment_trend.round(4),
        "existing_loans": existing_loans, "monthly_debt": monthly_debt.round(2), "DTI": DTI.round(4),
        "total_outstanding_debt": total_outstanding_debt.round(2), "mortgage_flag": mortgage_flag,
        "loan_amount": loan_amount, "loan_term_months": loan_term_months, "loan_purpose": loan_purpose,
        "interest_rate_requested": interest_rate_requested, "collateral_flag": collateral_flag,
        "loan_to_income": loan_to_income.round(4),
        "monthly_spending": monthly_spending.round(2), "spending_behaviour_index": spending_behaviour_index.round(4),
        "savings_ratio": savings_ratio.round(4), "transaction_volume": transaction_volume,
        "transaction_stability": transaction_stability.round(4), "avg_account_balance": avg_account_balance.round(2),
        "overdraft_count": overdraft_count, "loan_repayment_delay_days": loan_repayment_delay_days,
        "mobile_money_usage": mobile_money_usage.round(4), "mobile_payment_frequency": mobile_payment_frequency,
        "digital_wallet_balance": digital_wallet_balance.round(2),
        "utility_bill_consistency": utility_bill_consistency.round(4),
        "rent_payment_consistency": rent_payment_consistency.round(4), "ecommerce_txn_count": ecommerce_txn_count,
        "social_financial_score": social_financial_score.round(4), "account_category": account_category,
        "loan_approval_status": loan_approval_status,
        "default": default,
    })

    # inject missing values into a few columns (handled by Eq. 8)
    if missing_rate > 0:
        for col in ["annual_income", "years_employed", "monthly_spending", "mobile_money_usage",
                    "utility_bill_consistency", "education_level"]:
            mask = U(0, 1, n) < missing_rate
            df.loc[mask, col] = np.nan
    return df
