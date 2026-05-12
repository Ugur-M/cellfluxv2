"""Thin Weights & Biases wrapper used by the Stage 1 trainer.

The trainer always writes ``train.jsonl`` locally; wandb is an optional
parallel sink. The wrapper degrades to a no-op in three cases:

  - ``cfg["enabled"]`` is false,
  - ``cfg["mode"] == "disabled"``,
  - the ``wandb`` package can't be imported / wandb.init raises.

That way a misconfigured account or an offline node never blocks the
training run — the JSONL stays authoritative.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


class WandbRun:
    """A small ``wandb.init`` / ``wandb.log`` shim with a no-op fallback."""

    def __init__(
        self,
        cfg: Optional[dict[str, Any]],
        *,
        config: Optional[dict[str, Any]] = None,
        output_dir: Optional[str | Path] = None,
    ):
        self.active = False
        self._wandb = None
        self._run = None

        if not cfg or not bool(cfg.get("enabled", False)):
            return
        mode = str(cfg.get("mode", "online")).lower()
        if mode == "disabled":
            return

        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - import-time issues only
            print(f"[wandb] import failed: {exc}; continuing without wandb", flush=True)
            return

        kwargs: dict[str, Any] = {
            "project": cfg.get("project") or os.environ.get("WANDB_PROJECT", "cellfluxv2-stage1"),
            "mode": mode,
            "reinit": True,
        }
        if cfg.get("entity"):
            kwargs["entity"] = cfg["entity"]
        if cfg.get("run_name"):
            kwargs["name"] = cfg["run_name"]
        if cfg.get("tags"):
            kwargs["tags"] = list(cfg["tags"])
        if cfg.get("notes"):
            kwargs["notes"] = cfg["notes"]
        if config is not None:
            kwargs["config"] = config
        if output_dir is not None:
            kwargs["dir"] = str(Path(output_dir).resolve())

        try:
            self._run = wandb.init(**kwargs)
        except Exception as exc:
            print(f"[wandb] init failed: {exc}; continuing without wandb", flush=True)
            return

        self._wandb = wandb
        self.active = True
        try:
            url = getattr(self._run, "url", None)
        except Exception:
            url = None
        print(f"[wandb] run={getattr(self._run, 'name', '?')} url={url}", flush=True)

    def log(self, record: dict[str, Any], step: Optional[int] = None) -> None:
        if not self.active or self._wandb is None:
            return
        try:
            if step is None:
                self._wandb.log(record)
            else:
                self._wandb.log(record, step=int(step))
        except Exception as exc:  # pragma: no cover - network/runtime issues only
            print(f"[wandb] log failed at step={step}: {exc}", flush=True)

    def save(self, path: str | Path) -> None:
        if not self.active or self._wandb is None:
            return
        try:
            self._wandb.save(str(path), policy="now")
        except Exception as exc:  # pragma: no cover - network/runtime issues only
            print(f"[wandb] save failed for {path}: {exc}", flush=True)

    def finish(self) -> None:
        if not self.active or self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception as exc:  # pragma: no cover - network/runtime issues only
            print(f"[wandb] finish failed: {exc}", flush=True)


def flatten_for_wandb(record: dict[str, Any], parent: str = "") -> dict[str, Any]:
    """Flatten nested dicts into ``a/b/c`` keys so wandb panels are clean."""
    out: dict[str, Any] = {}
    for k, v in record.items():
        key = f"{parent}/{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_for_wandb(v, key))
        else:
            out[key] = v
    return out
