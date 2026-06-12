from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import yaml


ENV_CFG_FILENAMES = (
    "env.yaml",
    "drone.yaml",
    "obstacle.yaml",
    "dynamic_obstacle.yaml",
    "reward.yaml",
)


def load_yaml_cfg(path: str | None) -> dict[str, Any]:
    if path is None or not os.path.isfile(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def apply_env_cfg_dir(env_cfg: Any, cfg_dir: str | None) -> list[str]:
    if not cfg_dir:
        return []

    loaded_paths: list[str] = []
    for filename in ENV_CFG_FILENAMES:
        path = os.path.join(cfg_dir, filename)
        cfg = load_yaml_cfg(path)
        if not cfg:
            continue
        _apply_mapping(env_cfg, cfg, source=path)
        loaded_paths.append(path)

    if hasattr(env_cfg, "__post_init__"):
        env_cfg.__post_init__()
    return loaded_paths


def _apply_mapping(target: Any, cfg: Mapping[str, Any], source: str) -> None:
    for key, value in cfg.items():
        if value is None:
            continue

        if hasattr(target, key):
            current_value = getattr(target, key)
            if _is_nested_cfg(current_value):
                if not isinstance(value, Mapping):
                    raise ValueError(f"Expected mapping for '{key}' in {source}.")
                _apply_mapping(current_value, value, source)
            else:
                setattr(target, key, _coerce_value(current_value, value))
            continue

        raise ValueError(f"Unknown env config key '{key}' in {source}.")


def _is_nested_cfg(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _coerce_value(current_value: Any, value: Any) -> Any:
    if isinstance(current_value, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def dataclass_field_names(cfg: Any) -> set[str]:
    if not is_dataclass(cfg):
        return set()
    return {field.name for field in fields(cfg)}
