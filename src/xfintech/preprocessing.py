"""Data preprocessing and feature representation (Section 3.2, Eq. 7–11).

* Eq. 8  – mean imputation of missing numeric values
* Eq. 7  – z-score normalisation  x' = (x - mu) / sigma
* Eq. 9  – one-hot encoding of categorical attributes
* Eq. 10 – dense embeddings are learned inside the network (``FeatureTokenizer``)
* Eq. 11 – final representation Z = [x'_num ; e_cat]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TabularPreprocessor:
    target: str
    protected_attribute: str | None = None
    privileged_value: str | None = None
    drop_columns: list[str] = field(default_factory=list)
    max_categories: int = 30

    numeric_cols_: list[str] = field(default_factory=list)
    categorical_cols_: list[str] = field(default_factory=list)
    means_: dict = field(default_factory=dict)
    stds_: dict = field(default_factory=dict)
    categories_: dict = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    feature_groups_: list[str] = field(default_factory=list)  # original column of each output feature

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        cols = [c for c in df.columns if c != self.target and c not in self.drop_columns]
        self.numeric_cols_ = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_cols_ = [c for c in cols if c not in self.numeric_cols_]

        for c in self.numeric_cols_:
            x = df[c].astype(float)
            mu = float(x.mean())                       # Eq. 8
            sd = float(x.fillna(mu).std(ddof=0)) or 1.0
            self.means_[c], self.stds_[c] = mu, sd
        for c in self.categorical_cols_:
            vc = df[c].astype("string").fillna("__missing__").value_counts()
            self.categories_[c] = [str(v) for v in vc.index[: self.max_categories]]

        self.feature_names_, self.feature_groups_ = [], []
        for c in self.numeric_cols_:
            self.feature_names_.append(c)
            self.feature_groups_.append(c)
        for c in self.categorical_cols_:
            for v in self.categories_[c]:
                self.feature_names_.append(f"{c}={v}")
                self.feature_groups_.append(c)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        if self.numeric_cols_:
            num = np.column_stack([
                (df[c].astype(float).fillna(self.means_[c]).to_numpy() - self.means_[c]) / self.stds_[c]  # Eq. 7
                for c in self.numeric_cols_
            ])
            blocks.append(num)
        for c in self.categorical_cols_:
            s = df[c].astype("string").fillna("__missing__").astype(str)
            cats = self.categories_[c]
            codes = pd.Categorical(s, categories=cats).codes            # -1 for unseen categories
            onehot = np.zeros((len(df), len(cats)), dtype=np.float32)   # Eq. 9
            rows = np.where(codes >= 0)[0]
            onehot[rows, codes[rows]] = 1.0
            blocks.append(onehot)
        return np.concatenate(blocks, axis=1).astype(np.float32)        # Eq. 11

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def labels(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.target].astype(int).to_numpy()

    def protected(self, df: pd.DataFrame) -> np.ndarray:
        """Binary protected-group indicator A (1 = privileged)."""
        if not self.protected_attribute or self.protected_attribute not in df:
            return np.zeros(len(df), dtype=np.float32)
        col = df[self.protected_attribute]
        if self.privileged_value is None:
            self.privileged_value = str(col.mode().iloc[0])
        return (col.astype("string") == str(self.privileged_value)).to_numpy(dtype=np.float32)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        state = {k: getattr(self, k) for k in self.__dataclass_fields__}
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        state = json.loads(Path(path).read_text())
        return cls(**state)


def aggregate_by_group(values: np.ndarray, groups: list[str]) -> pd.Series:
    """Sum per-feature importances of one-hot columns back to the original attribute."""
    s = pd.Series(np.asarray(values, dtype=float), index=groups)
    return s.groupby(level=0, sort=False).sum().sort_values(ascending=False)
