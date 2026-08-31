#!/usr/bin/env python3
"""Slice the first N rows from the real geo3k parquet dataset for debugging.

Usage:
    # Default: 2 train + 1 val rows
    python3 slice_small_data.py

    # Custom sizes
    python3 slice_small_data.py --train 4 --val 2 --save_dir /tmp/geo3k_tiny

    # Override source paths
    python3 slice_small_data.py \
        --train_file /path/to/train.parquet \
        --val_file /path/to/test.parquet \
        --train 2 --val 1
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice tiny dataset from real geo3k parquet")
    parser.add_argument("--train_file", default="/opt/huawei/dataset/ccb_data/data/omni/geo3k/train.parquet")
    parser.add_argument("--val_file", default="/opt/huawei/dataset/ccb_data/data/omni/geo3k/test.parquet")
    parser.add_argument("--train", type=int, default=2, help="Number of train rows to slice")
    parser.add_argument("--val", type=int, default=1, help="Number of val rows to slice")
    parser.add_argument("--save_dir", default="/tmp/geo3k_tiny")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train = pd.read_parquet(args.train_file).head(args.train)
    test = pd.read_parquet(args.val_file).head(args.val)

    train_path = os.path.join(args.save_dir, "train.parquet")
    test_path = os.path.join(args.save_dir, "test.parquet")
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    print(f"Saved: {train_path}  ({len(train)} rows)")
    print(f"       {test_path}  ({len(test)} rows)")
    print(f"Columns: {list(train.columns)}")
    print()
    print(f"export TRAIN_FILE={train_path}")
    print(f"export VAL_FILE={test_path}")


if __name__ == "__main__":
    main()