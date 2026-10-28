"""Configuration helpers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml", overrides: dict | None = None) -> dict:
    """Load a YAML config and apply nested ``overrides`` (dict of dicts)."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_cli_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["model.dropout=0.4", "training.epochs=5"]`` into a nested dict."""
    result: dict[str, Any] = {}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        node = result
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return result
