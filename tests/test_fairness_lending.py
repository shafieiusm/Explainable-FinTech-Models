import numpy as np

from xfintech.evaluate import classification_metrics
from xfintech.fairness import (demographic_parity_difference, disparate_impact_ratio,
                               equalized_odds_difference)
from xfintech.lending import (LendingEngine, behaviour_evolution, credit_utilisation, debt_to_income,
                              dynamic_risk_trajectory, dynamic_risk_update)


def test_demographic_parity_and_equalized_odds():
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_hat = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    assert np.isclose(demographic_parity_difference(y_hat, a), 0.25)
    assert np.isclose(disparate_impact_ratio(y_hat, a), 0.5)
    assert np.isclose(equalized_odds_difference(y, y_hat, a), 0.5)


def test_ratios():
    assert np.allclose(debt_to_income([500, 0], [2000, 0]), [0.25, 0])        # Eq. 53
    assert np.allclose(credit_utilisation([300], [1000]), [0.3])              # Eq. 54


def test_lending_engine_rules():
    e = LendingEngine(base_rate=4, alpha=18, approval_threshold=0.6)
    crs = np.array([0.1, 0.45, 0.8])
    assert np.allclose(e.interest_rate(crs), [5.8, 12.1, 18.4])              # Eq. 55
    assert list(e.decision(crs)) == ["Approved", "Approved", "Rejected"]      # Eq. 56
    assert list(e.risk_band(crs)) == ["Low Risk", "Medium Risk", "High Risk"]
    t = e.recommend(["B1", "B2", "B3"], crs, income=[9000, 4000, 2000], spending=[2000, 2500, 1900])
    assert t.loc[0, "recommendation_score"] > t.loc[2, "recommendation_score"]  # Eq. 57


def test_dynamic_updates():
    assert np.isclose(dynamic_risk_update(0.5, 1.0, 0.7), 0.65)              # Eq. 42
    traj = dynamic_risk_trajectory([0.4, 0.8, 0.8, 0.2], lam=0.3)
    assert traj[2] > traj[1] > traj[0] and traj[3] < traj[2]
    b = behaviour_evolution([1, 1, 0, 0], eta=0.5)                           # Eq. 58
    assert np.allclose(b, [1, 1, 0.5, 0.25])


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == 100 and m["auc_roc"] == 100 and np.isclose(m["mcc"], 1.0)


def test_explain_helpers_without_tensorflow():
    import pandas as pd

    from xfintech.explain import ensemble_importance, global_importance, permutation_importance

    phi = np.array([[0.2, -0.1, 0.1], [-0.4, 0.1, 0.1]])
    gi = global_importance(phi, ["a", "b", "b"])               # Eq. 44 with one-hot grouping
    assert np.isclose(gi["a"], 0.3) and np.isclose(gi["b"], 0.1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = (X[:, 0] > 0).astype(int)
    pi = permutation_importance(lambda z: 1 / (1 + np.exp(-5 * z[:, 0])), X, y, ["x0", "x1"], n_repeats=2)
    assert pi.index[0] == "x0" and pi["x1"] == 0
    ens = ensemble_importance(pd.Series({"x0": 2.0, "x1": 1.0}), pd.Series({"x0": 1.0, "x1": 1.0}))
    assert np.isclose(ens.sum(), 1) and ens.index[0] == "x0"
