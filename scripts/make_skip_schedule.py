#!/usr/bin/env python3
"""
make_skip_schedule.py
---------------------

Reads an across-step cosine similarity CSV (from Step 2)
and builds a skip schedule JSON for Dream (and later LLaDA).

Usage:
    python scripts/make_skip_schedule.py \
        --csv /content/drive/MyDrive/Fast-dLLM-logs/dream_baseline/across_steps_cosine.csv \
        --tau 0.98 \
        --out /content/drive/MyDrive/Fast-dLLM-logs/dream_baseline/skip_schedule_tau0.98.json

Output JSON example:
{
  "num_steps": 16,
  "layers": 28,
  "skip": {
    "0": [],
    "1": [3, 5, 12, 21],
    "2": [2, 8, 20],
    ...
  }
}
"""

import argparse
import csv
import json


def build_skip_schedule(csv_path: str, tau: float):
    """Builds skip dictionary from across-step cosine CSV."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Identify layer columns (e.g., layer_0, layer_1, ...)
        layer_cols = [i for i, h in enumerate(header) if h.startswith("layer_")]
        num_layers = len(layer_cols)

        skip = {}
        for row in reader:
            step = int(row[0])
            skip_layers = []
            if step > 0:  # Step 0 can't reuse anything
                for j, col in enumerate(layer_cols):
                    try:
                        val = float(row[col])
                        if val >= tau:
                            skip_layers.append(j)
                    except ValueError:
                        continue
            skip[step] = skip_layers

    return {"num_steps": len(skip), "layers": num_layers, "skip": skip}


def main():
    ap = argparse.ArgumentParser(description="Generate skip schedule JSON from across-step cosine CSV.")
    ap.add_argument("--csv", required=True, help="Path to across_steps_cosine.csv from Step 2.")
    ap.add_argument("--tau", type=float, default=0.98, help="Similarity threshold for skipping (default: 0.98).")
    ap.add_argument("--out", required=True, help="Output JSON file path.")
    args = ap.parse_args()

    schedule = build_skip_schedule(args.csv, args.tau)

    with open(args.out, "w") as f:
        json.dump(schedule, f, indent=2)

    print(f"[make_skip_schedule] Wrote schedule to {args.out}")
    print(f"  τ = {args.tau}")
    print(f"  Steps: {schedule['num_steps']}, Layers: {schedule['layers']}")
    for s, layers in schedule["skip"].items():
        if layers:
            print(f"    step {s}: skip {len(layers)} layers ({layers[:6]}{'...' if len(layers)>6 else ''})")


if __name__ == "__main__":
    main()
