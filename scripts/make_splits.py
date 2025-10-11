#!/usr/bin/env python3
# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Deterministic, stratified K-fold split generator (stdlib-only).

- Inputs: CSV manifest with header: id,label (stratify_by can override 'label')
- Algorithm: For each label, seeded shuffle then round-robin assignment across folds.
- Outputs: For each fold i in [0..k-1], write JSON arrays:
  - fold_i_train.json
  - fold_i_val.json

Usage:
  python scripts/make_splits.py --input_manifest ./manifests/dataset_manifest.csv --output_dir ./splits --k_folds 5 --seed 42 --stratify_by label

Defaults mirror configs/cv.yaml:
  k_folds: 5
  seed: 42
  input_manifest: "./manifests/dataset_manifest.csv"
  output_dir: "./splits"
  stratify_by: "label"

No external dependencies required.
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make deterministic stratified K-fold splits from a CSV manifest.")
    p.add_argument("--input_manifest", "-i", type=str, default="./manifests/dataset_manifest.csv", help="Path to CSV manifest with header 'id' and label column (default 'label').")
    p.add_argument("--output_dir", "-o", type=str, default="./splits", help="Directory to write JSON splits.")
    p.add_argument("--k_folds", "-k", type=int, default=5, help="Number of folds (>=2).")
    p.add_argument("--seed", "-s", type=int, default=42, help="Random seed for shuffling within label groups.")
    p.add_argument("--stratify_by", "-y", type=str, default="label", help="Column name to stratify by (default: 'label').")
    return p.parse_args()


def read_manifest(path: str, stratify_by: str) -> Tuple[Dict[str, List[str]], int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input manifest not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV header is required (expected at least 'id' and the stratify_by column).")
        header_map = {fn.strip().lower(): fn for fn in reader.fieldnames}
        id_key = header_map.get("id")
        label_key = header_map.get(stratify_by.strip().lower())
        if id_key is None:
            raise ValueError("CSV is missing required 'id' column.")
        if label_key is None:
            raise ValueError(f"CSV is missing required stratify_by column: '{stratify_by}'")

        groups: Dict[str, List[str]] = {}
        seen_ids: set = set()
        total_rows = 0
        for idx, row in enumerate(reader, start=2):  # data starts after header (line 1)
            total_rows += 1
            sample_id = (row.get(id_key) or "").strip()
            label = (row.get(label_key) or "").strip()
            if sample_id == "":
                raise ValueError(f"Row {idx}: empty 'id' value.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate id encountered: '{sample_id}' (row {idx}).")
            seen_ids.add(sample_id)
            groups.setdefault(label, []).append(sample_id)

    return groups, total_rows


def stratified_round_robin(groups: Dict[str, List[str]], k_folds: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    folds: List[List[str]] = [[] for _ in range(k_folds)]
    for label in sorted(groups.keys()):
        ids = list(groups[label])
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            folds[i % k_folds].append(sid)
    return folds


def write_folds(folds: List[List[str]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Deterministic base ordering for training sets
    all_ids_sorted = sorted({sid for fold in folds for sid in fold})
    for i, val_ids in enumerate(folds):
        val_sorted = sorted(val_ids)
        val_set = set(val_sorted)
        train_sorted = [sid for sid in all_ids_sorted if sid not in val_set]

        train_path = os.path.join(output_dir, f"fold_{i}_train.json")
        val_path = os.path.join(output_dir, f"fold_{i}_val.json")
        with open(train_path, "w", encoding="utf-8") as ft:
            json.dump(train_sorted, ft, indent=2)
        with open(val_path, "w", encoding="utf-8") as fv:
            json.dump(val_sorted, fv, indent=2)
        print(f"fold {i}: train={len(train_sorted)} val={len(val_sorted)}")


def main() -> None:
    args = parse_args()
    if args.k_folds < 2:
        print("error: --k_folds must be >= 2", file=sys.stderr)
        sys.exit(2)
    try:
        groups, total_rows = read_manifest(args.input_manifest, args.stratify_by)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    total_ids = sum(len(v) for v in groups.values())
    if total_ids == 0:
        print("warning: no rows found in manifest; nothing to split.", file=sys.stderr)
        sys.exit(0)

    folds = stratified_round_robin(groups, args.k_folds, args.seed)
    print(f"loaded {total_ids} samples across {len(groups)} labels; writing splits to: {args.output_dir}")
    write_folds(folds, args.output_dir)


if __name__ == "__main__":
    main()