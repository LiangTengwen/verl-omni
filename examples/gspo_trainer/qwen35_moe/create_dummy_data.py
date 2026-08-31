#!/usr/bin/env python3
"""Generate a tiny dummy parquet dataset for debugging the Qwen3.5 MoE GSPO
Megatron training pipeline on NPU.

Usage:
    python3 create_dummy_data.py --save_dir /path/to/dummy_geo3k

Output schema (mirrors the geo3k / mmk12 format):
    data_source, prompt, images, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from PIL import Image


def _write_dummy_image(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (64, 64), color=(42, 128, 200)).save(path)


def _make_row(
    idx: int,
    image_path: str,
    split: str,
    question: str,
    answer: str,
) -> dict:
    return {
        "data_source": "dummy_geo3k",
        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ],
        "images": [image_path],
        "ability": "geo_qa",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "id": f"dummy_{split}_{idx}",
            "dataset": "dummy",
            "raw_question": question,
            "raw_answer": answer,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny dummy geo3k parquet for debugging")
    parser.add_argument(
        "--save_dir",
        default=os.path.expanduser("~/data/dummy_geo3k"),
        help="Output directory for the parquet files and media",
    )
    parser.add_argument("--train_size", type=int, default=2, help="Number of train rows")
    parser.add_argument("--val_size", type=int, default=1, help="Number of val rows")
    args = parser.parse_args()

    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    media_dir = os.path.join(save_dir, "media", "images")
    os.makedirs(media_dir, exist_ok=True)

    # Write a single dummy image reused by all samples.
    dummy_image = os.path.join(media_dir, "dummy.png")
    _write_dummy_image(dummy_image)

    train_rows = []
    for i in range(args.train_size):
        train_rows.append(
            _make_row(
                idx=i,
                image_path=dummy_image,
                split="train",
                question=f"Debug question {i}: What is shown in this image?",
                answer=f"Debug answer {i}",
            )
        )

    val_rows = []
    for i in range(args.val_size):
        val_rows.append(
            _make_row(
                idx=i,
                image_path=dummy_image,
                split="val",
                question=f"Debug val question {i}: Describe this picture.",
                answer=f"Debug val answer {i}",
            )
        )

    pd.DataFrame(train_rows).to_parquet(os.path.join(save_dir, "train.parquet"), index=False)
    pd.DataFrame(val_rows).to_parquet(os.path.join(save_dir, "test.parquet"), index=False)

    print(f"Dummy data written to {save_dir}")
    print(f"  train: {args.train_size} rows -> {save_dir}/train.parquet")
    print(f"  val:   {args.val_size} rows -> {save_dir}/test.parquet")
    print(f"  image: {dummy_image}")
    print()
    print("Usage:")
    print(f"  export TRAIN_FILE={save_dir}/train.parquet")
    print(f"  export VAL_FILE={save_dir}/test.parquet")


if __name__ == "__main__":
    main()