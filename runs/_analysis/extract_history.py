"""Extract metric history from local wandb .wandb run files."""
import json
import sys
from pathlib import Path

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2

def extract(wandb_file: Path) -> list[dict]:
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    rows = []
    bad = 0
    while True:
        try:
            raw = ds.scan_record()
        except (IndexError, AssertionError):
            bad += 1
            if bad > 1000:
                break
            continue
        if raw is None:
            break
        # raw is (num, record_bytes)
        try:
            _num, payload = raw
        except (TypeError, ValueError):
            payload = raw
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(payload)
        except Exception:
            continue
        if rec.HasField("history"):
            row = {}
            for item in rec.history.item:
                if item.key:
                    key = item.key
                else:
                    key = ".".join(list(item.nested_key))
                try:
                    row[key] = json.loads(item.value_json)
                except Exception:
                    row[key] = item.value_json
            rows.append(row)
    return rows

if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    rows = extract(src)
    out.write_text(json.dumps(rows))
    print(f"{src.name}: {len(rows)} rows  -> {out}")
