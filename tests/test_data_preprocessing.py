import numpy as np

from xfintech.data import make_synthetic_dataset, stratified_kfold_indices, stratified_split
from xfintech.preprocessing import TabularPreprocessor, aggregate_by_group


def test_synthetic_schema_and_default_rate():
    df = make_synthetic_dataset(n=20_000, seed=0)
    assert df.shape[1] - 2 == 51                    # 51 attributes + id + target
    assert abs(df["default"].mean() - 0.1371) < 0.01
    assert df["credit_score"].between(300, 850).all()


def test_split_is_stratified_and_disjoint():
    df = make_synthetic_dataset(n=10_000, seed=1)
    tr, va, te = stratified_split(df, "default")
    assert len(tr) + len(va) + len(te) == len(df)
    assert set(tr.borrower_id).isdisjoint(te.borrower_id)
    assert abs(tr["default"].mean() - te["default"].mean()) < 0.01


def test_preprocessor_normalises_imputes_and_encodes(tmp_path):
    df = make_synthetic_dataset(n=5_000, seed=2)
    pre = TabularPreprocessor("default", "gender", "Male", ["borrower_id", "loan_approval_status"])
    X = pre.fit_transform(df)
    assert not np.isnan(X).any()                                   # Eq. 8
    num = X[:, : len(pre.numeric_cols_)]
    assert np.allclose(num.mean(0), 0, atol=1e-4)                  # Eq. 7
    onehot = X[:, len(pre.numeric_cols_):]
    assert set(np.unique(onehot)) <= {0.0, 1.0}                    # Eq. 9
    assert len(pre.feature_names_) == X.shape[1]
    pre.save(tmp_path / "p.json")
    assert np.allclose(TabularPreprocessor.load(tmp_path / "p.json").transform(df), X)


def test_kfold_covers_all_rows_once():
    y = np.r_[np.zeros(90), np.ones(10)].astype(int)
    seen = np.concatenate([te for _, te in stratified_kfold_indices(y, 5)])
    assert sorted(seen) == list(range(100))


def test_aggregate_by_group():
    s = aggregate_by_group([0.1, 0.2, 0.3], ["a", "b", "b"])
    assert s["b"] == 0.5 and s.index[0] == "b"
