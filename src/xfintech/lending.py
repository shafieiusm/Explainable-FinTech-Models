"""Personalised lending recommendation engine and dynamic risk adaptation
(Sections 3.7 and 3.10, Eq. 41–42 and 53–58).  Pure NumPy / pandas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ratios
def debt_to_income(monthly_debt, monthly_income):
    """DTI = MonthlyDebt / MonthlyIncome  (Eq. 53)."""
    monthly_income = np.asarray(monthly_income, dtype=float)
    return np.divide(monthly_debt, monthly_income, out=np.zeros_like(monthly_income), where=monthly_income > 0)


def credit_utilisation(credit_used, credit_limit):
    """CUR = CreditUsed / CreditLimit  (Eq. 54)."""
    credit_limit = np.asarray(credit_limit, dtype=float)
    return np.divide(credit_used, credit_limit, out=np.zeros_like(credit_limit), where=credit_limit > 0)


# ---------------------------------------------------------- risk scores
def linear_credit_risk_score(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """CRS_i = Σ_j w_j x_ij  (Eq. 41) — squashed to [0, 1] with a sigmoid."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(X) @ np.asarray(w))))


def dynamic_risk_update(prev_crs: float, current_risk: float, lam: float = 0.7) -> float:
    """CRS_t = λ CRS_{t−1} + (1 − λ) f(x_t)  (Eq. 42)."""
    return lam * prev_crs + (1.0 - lam) * current_risk


def dynamic_risk_trajectory(monthly_risk: list[float] | np.ndarray, lam: float = 0.7, init: float | None = None):
    """Apply Eq. 42 over a sequence of monthly model scores f(x_t)."""
    monthly_risk = np.asarray(monthly_risk, dtype=float)
    out = np.zeros_like(monthly_risk)
    prev = monthly_risk[0] if init is None else init
    for t, r in enumerate(monthly_risk):
        prev = dynamic_risk_update(prev, r, lam)
        out[t] = prev
    return out


def behaviour_evolution(x_seq: np.ndarray, eta: float = 0.8) -> np.ndarray:
    """B_t = η B_{t−1} + (1 − η) x_t  (Eq. 58) — exponentially smoothed behaviour profile."""
    x_seq = np.asarray(x_seq, dtype=float)
    out = np.zeros_like(x_seq)
    b = x_seq[0]
    for t in range(len(x_seq)):
        b = eta * b + (1 - eta) * x_seq[t]
        out[t] = b
    return out


# ------------------------------------------------------- lending engine
@dataclass
class LendingEngine:
    base_rate: float = 4.0            # R_b (%)
    alpha: float = 18.0               # α
    approval_threshold: float = 0.60  # τ
    low_band: float = 0.30
    high_band: float = 0.60
    prs_weights: tuple = (-0.6, 0.25, -0.15)  # ω1 (CRS), ω2 (income), ω3 (spending)

    @classmethod
    def from_config(cls, cfg: dict) -> "LendingEngine":
        bands = cfg.get("risk_bands", {})
        return cls(base_rate=cfg.get("base_rate", 4.0), alpha=cfg.get("alpha", 18.0),
                   approval_threshold=cfg.get("approval_threshold", 0.6),
                   low_band=bands.get("low", 0.3), high_band=bands.get("high", 0.6),
                   prs_weights=tuple(cfg.get("prs_weights", (-0.6, 0.25, -0.15))))

    def interest_rate(self, crs):
        """R_i = R_b + α · CRS_i  (Eq. 55)."""
        return self.base_rate + self.alpha * np.asarray(crs, dtype=float)

    def decision(self, crs):
        """Approved if CRS_i < τ, otherwise Rejected  (Eq. 56)."""
        return np.where(np.asarray(crs) < self.approval_threshold, "Approved", "Rejected")

    def risk_band(self, crs):
        crs = np.asarray(crs, dtype=float)
        return np.where(crs < self.low_band, "Low Risk", np.where(crs < self.high_band, "Medium Risk", "High Risk"))

    def recommendation_score(self, crs, income, spending):
        """PRS_i = ω1 CRS_i + ω2 Income_i + ω3 Spending_i  (Eq. 57).

        Income and spending are min–max scaled within the batch so the weights
        are comparable; a higher PRS means a more favourable offer.
        """
        def mm(v):
            v = np.asarray(v, dtype=float)
            rng = np.nanmax(v) - np.nanmin(v)
            return (v - np.nanmin(v)) / rng if rng > 0 else np.zeros_like(v)

        w1, w2, w3 = self.prs_weights
        return w1 * np.asarray(crs, float) + w2 * mm(income) + w3 * mm(spending)

    def recommend(self, borrower_ids, crs, income=None, spending=None) -> pd.DataFrame:
        crs = np.asarray(crs, dtype=float)
        out = pd.DataFrame({
            "borrower_id": list(borrower_ids),
            "credit_risk_score": crs.round(4),
            "risk_band": self.risk_band(crs),
            "interest_rate_pct": self.interest_rate(crs).round(2),
            "decision": self.decision(crs),
        })
        if income is not None and spending is not None:
            out["recommendation_score"] = self.recommendation_score(crs, income, spending).round(4)
            out["offer"] = np.select(
                [out["decision"].eq("Rejected"), out["risk_band"].eq("Low Risk"), out["risk_band"].eq("Medium Risk")],
                ["Decline — suggest secured / smaller loan", "Pre-approved — premium rate",
                 "Approve with standard terms"],
                default="Approve with risk-adjusted pricing",
            )
        return out
