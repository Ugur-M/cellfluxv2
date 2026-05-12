"""Checkpoint save / load with atomic writes.

A "checkpoint" here is a single ``torch.save`` blob containing:
  - ``model_state``   : model.state_dict()
  - ``step``          : training step counter (int, >= 0)
  - ``epoch``         : epoch counter (int, >= 0)
  - ``config``        : free-form config dict
  - ``extra``         : free-form extra dict
  - ``optimizer_state`` (optional, only when an optimizer is passed)

``save_checkpoint`` writes to ``path.tmp`` then renames over ``path``
so a crashed save can never leave a half-written file at the target.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

REQUIRED_KEYS: tuple[str, ...] = ("model_state", "step", "epoch", "config", "extra")


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int,
    epoch: int,
    config: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Write a checkpoint atomically.

    Validates ``step``, ``epoch``, ``model``, optional ``optimizer``;
    creates parent directories; writes to ``<path>.tmp`` and then
    ``os.replace`` over the final path.
    """
    if not isinstance(model, nn.Module):
        raise ValueError(f"model must be an nn.Module; got {type(model).__name__}")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer or None; "
            f"got {type(optimizer).__name__}"
        )
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError(f"step must be an int; got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be >= 0; got {step}")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise ValueError(f"epoch must be an int; got {type(epoch).__name__}")
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0; got {epoch}")
    if config is not None and not isinstance(config, dict):
        raise ValueError(f"config must be a dict or None; got {type(config).__name__}")
    if extra is not None and not isinstance(extra, dict):
        raise ValueError(f"extra must be a dict or None; got {type(extra).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obj: dict[str, Any] = {
        "model_state": model.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "config": dict(config) if config is not None else {},
        "extra": dict(extra) if extra is not None else {},
    }
    if optimizer is not None:
        obj["optimizer_state"] = optimizer.state_dict()

    tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else Path(str(path) + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` (and optionally ``optimizer``).

    Returns the metadata dict ``{step, epoch, config, extra}``. Raises
    ``FileNotFoundError`` if the file is missing and ``ValueError`` if
    any required key is absent.
    """
    if not isinstance(model, nn.Module):
        raise ValueError(f"model must be an nn.Module; got {type(model).__name__}")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer or None; "
            f"got {type(optimizer).__name__}"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    obj = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(
            f"checkpoint at {path} must be a dict; got {type(obj).__name__}"
        )
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"checkpoint at {path} missing required keys: {missing}")

    model.load_state_dict(obj["model_state"])
    if optimizer is not None and "optimizer_state" in obj:
        optimizer.load_state_dict(obj["optimizer_state"])

    return {
        "step": obj["step"],
        "epoch": obj["epoch"],
        "config": obj.get("config", {}),
        "extra": obj.get("extra", {}),
    }
