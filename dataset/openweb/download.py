#!/usr/bin/env python3
"""
Download and prepare OpenWebText for LM_Analysis.

OpenWebText (Skylion007) is a reproduction of the GPT-2 training corpus,
scraped from URLs shared on Reddit with ≥3 upvotes.  ~8GB uncompressed text.

Requirements:
    pip install datasets huggingface_hub

Output files (same directory as this script):
    train   ~8M documents, ~8GB
    valid   10,000 documents held out
    test    10,000 documents held out

Usage:
    python download.py
    python download.py --cache_dir /scratch/hf_cache   # custom HF cache
    python download.py --workers 8                     # parallel tokenisation
"""

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description="Download OpenWebText")
    p.add_argument(
        "--cache_dir",
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for dataset streaming (default: 4)",
    )
    p.add_argument(
        "--valid_size",
        type=int,
        default=10_000,
        help="Number of documents to hold out for valid (default: 10000)",
    )
    p.add_argument(
        "--test_size",
        type=int,
        default=10_000,
        help="Number of documents to hold out for test (default: 10000)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def write_split(documents, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            # One document per line; internal newlines collapsed to spaces.
            line = doc.replace("\n", " ").strip()
            if line:
                f.write(line + "\n")
    print(f"  wrote {len(documents):,} docs → {path}")


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install HuggingFace datasets:  pip install datasets")

    print("Downloading OpenWebText from HuggingFace (Skylion007/openwebtext)...")
    print("  ~8GB download, may take a while on first run.\n")

    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        cache_dir=args.cache_dir,
        num_proc=args.workers,
    )

    print(f"Total documents: {len(ds):,}")

    # Deterministic shuffle then split
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    test_idx = set(indices[: args.test_size])
    valid_idx = set(indices[args.test_size : args.test_size + args.valid_size])

    train_docs, valid_docs, test_docs = [], [], []
    print("Splitting into train / valid / test ...")
    for i, example in enumerate(ds):
        text = example["text"]
        if i in test_idx:
            test_docs.append(text)
        elif i in valid_idx:
            valid_docs.append(text)
        else:
            train_docs.append(text)
        if (i + 1) % 500_000 == 0:
            print(f"  processed {i + 1:,} / {len(ds):,}")

    print("\nWriting splits ...")
    write_split(train_docs, HERE / "train")
    write_split(valid_docs, HERE / "valid")
    write_split(test_docs, HERE / "test")

    print("\nDone.  To analyse with LM_Analysis, first register the paths in")
    print("  src/data/loader.py  under the 'openweb' key, then run:")
    print("  python run_all.py --data dataset/openweb/.train --words 1 \\")
    print("      --threads 4 --overlap 1 --cutoff 10000 --save_path openweb_train")


if __name__ == "__main__":
    main()
