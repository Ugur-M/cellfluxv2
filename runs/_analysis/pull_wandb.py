"""Pull full history of runs from the wandb cloud."""
import json
import sys
from pathlib import Path
import wandb

RUNS = {
    "stage1_baseline": "immoustafa0-city-university-of-london/cellfluxv2-stage1/hyeekygn",
    "stage1_cond_balance": "immoustafa0-city-university-of-london/cellfluxv2-stage1/xhl7cib2",
    "stage2_smoke": "immoustafa0-city-university-of-london/cellfluxv2-stage2/95qevp4r",
}

api = wandb.Api(timeout=30)
for label, path in RUNS.items():
    run = api.run(path)
    print(f"{label}: {run.name} state={run.state} step={run.summary.get('_step')}")
    hist = run.history(samples=10000, pandas=False)
    out = Path(f"{label}_cloud.json")
    out.write_text(json.dumps(hist))
    print(f"  -> {out} ({len(hist)} rows, keys: {len(hist[0]) if hist else 0})")
