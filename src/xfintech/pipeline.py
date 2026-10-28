"""End-to-end experiment helpers shared by the scripts."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset, make_synthetic_dataset, stratified_split
from .evaluate import best_f1_threshold, classification_metrics
from .fairness import fairness_report
from .preprocessing import TabularPreprocessor


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_dataframe(cfg: dict, synthetic_if_missing: bool = True, n_synthetic: int = 300_000) -> pd.DataFrame:
    path = Path(cfg["data"]["path"])
    if path.exists():
        return load_dataset(path)
    if not synthetic_if_missing:
        raise FileNotFoundError(path)
    print(f"[data] {path} not found — generating {n_synthetic:,} synthetic borrower records")
    return make_synthetic_dataset(n=n_synthetic, seed=cfg["data"].get("seed", 42))


def prepare_splits(df: pd.DataFrame, cfg: dict) -> dict:
    dcfg = cfg["data"]
    df = df.drop(columns=[c for c in dcfg.get("id_columns", []) if c in df], errors="ignore")
    sp = dcfg.get("split", {})
    tr, va, te = stratified_split(df, dcfg["target"], sp.get("train", 0.7), sp.get("val", 0.15), dcfg.get("seed", 42))
    pre = TabularPreprocessor(target=dcfg["target"], protected_attribute=dcfg.get("protected_attribute"),
                              privileged_value=dcfg.get("privileged_value"),
                              drop_columns=[c for c in dcfg.get("drop_columns", []) if c in df])
    pre.fit(tr)
    out = {"pre": pre, "frames": {"train": tr, "val": va, "test": te}}
    for name, frame in out["frames"].items():
        out[f"X_{name}"] = pre.transform(frame)
        out[f"y_{name}"] = pre.labels(frame)
        out[f"a_{name}"] = pre.protected(frame)
    return out


def run_experiment(cfg: dict, splits: dict, verbose: int = 1) -> dict:
    """Build, train and evaluate one configuration.  Returns model + metrics."""
    from .model import build_model, predict_proba
    from .train import Trainer

    set_seed(cfg["data"].get("seed", 42))
    model = build_model(splits["X_train"].shape[1], cfg["model"])
    trainer = Trainer(model, cfg["training"], cfg.get("fairness"), verbose=verbose)
    history = trainer.fit(splits["X_train"], splits["y_train"], splits["a_train"],
                          splits["X_val"], splits["y_val"], splits["a_val"])

    p_val = predict_proba(model, splits["X_val"])
    thr_mode = cfg.get("evaluation", {}).get("threshold", "f1")
    thr = best_f1_threshold(splits["y_val"], p_val) if thr_mode == "f1" else float(thr_mode)
    p_test = predict_proba(model, splits["X_test"])
    metrics = classification_metrics(splits["y_test"], p_test, thr)
    fair = fairness_report(splits["y_test"], (p_test >= thr).astype(int), splits["a_test"])
    return {"model": model, "history": history, "metrics": metrics, "fairness": fair,
            "p_test": p_test, "threshold": thr}


def update_results(path: str | Path, **sections) -> dict:
    """Merge ``sections`` into a JSON results file (used by make_figures.py)."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(sections)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=float))
    return data
