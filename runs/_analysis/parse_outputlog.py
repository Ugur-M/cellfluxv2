"""Parse `[step N/M epoch E] k=v k=v ...` lines from output.log into a row list."""
import re
import sys
import json
from pathlib import Path

LINE_RE = re.compile(r"^\[step (\d+)/(\d+) epoch (\d+)\]\s+(.*)$")
KV_RE = re.compile(r"(\w+)=([\-0-9eE.+nan]+)")

def parse(log: Path):
    rows = []
    for line in log.read_text().splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        row = {"step": int(m.group(1)), "max_steps": int(m.group(2)), "epoch": int(m.group(3))}
        for k, v in KV_RE.findall(m.group(4)):
            try:
                row[k] = float(v)
            except ValueError:
                pass
        rows.append(row)
    return rows

if __name__ == "__main__":
    rows = parse(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(rows))
    print(f"{sys.argv[1]}: {len(rows)} rows -> {sys.argv[2]}")
