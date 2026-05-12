#!/usr/bin/env python3
"""
Download and prepare English Wikipedia for LM_Analysis.

Downloads the latest Wikipedia XML dump, extracts clean article text using
WikiExtractor, then builds train / valid / test splits in the same plain-text
format as WikiText-103 (one article per line, pre-tokenised).

Requirements:
    pip install wikiextractor huggingface_hub datasets

  OR use the HuggingFace Wikipedia dataset (faster, no XML parsing):
    python download.py --source hf

Output files (same directory as this script):
    train   ~90% of articles
    valid   ~5%  of articles
    test    ~5%  of articles

Usage:
    # Recommended — HuggingFace preprocessed dump (20241101 snapshot):
    python download.py --source hf

    # Full XML dump (gives you control over preprocessing):
    python download.py --source dump

    # Custom snapshot date (only relevant for --source hf):
    python download.py --source hf --date 20231101
"""

import argparse
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"


def parse_args():
    p = argparse.ArgumentParser(description="Download English Wikipedia")
    p.add_argument(
        "--source",
        choices=["hf", "dump"],
        default="hf",
        help="'hf' = HuggingFace preprocessed (fast); 'dump' = raw XML (slow)",
    )
    p.add_argument(
        "--date",
        default="20241101",
        help="Wikipedia snapshot date for HF source, e.g. 20231101 (default: 20241101)",
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help="HuggingFace cache directory",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)",
    )
    p.add_argument(
        "--valid_frac",
        type=float,
        default=0.05,
        help="Fraction of articles for valid split (default: 0.05)",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.05,
        help="Fraction of articles for test split (default: 0.05)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^=+\s.*\s=+$", re.MULTILINE)
_MULTI_SPACE = re.compile(r" {2,}")


def clean_article(text):
    """Strip section headers and collapse whitespace; return single-line string."""
    text = _SECTION_RE.sub("", text)
    text = text.replace("\n", " ").strip()
    text = _MULTI_SPACE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# HuggingFace path
# ---------------------------------------------------------------------------


def download_hf(args):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install HuggingFace datasets:  pip install datasets")

    print(f"Loading Wikipedia ({args.date}, en) from HuggingFace ...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        f"{args.date}.en",
        split="train",
        cache_dir=args.cache_dir,
        num_proc=args.workers,
    )
    print(f"  {len(ds):,} articles downloaded.")
    return [clean_article(ex["text"]) for ex in ds if ex["text"].strip()]


# ---------------------------------------------------------------------------
# Raw XML dump path
# ---------------------------------------------------------------------------


def download_dump(args):
    import subprocess
    import tempfile

    dump_path = HERE / "enwiki-latest-pages-articles.xml.bz2"
    extract_dir = HERE / "extracted"

    # Download
    if not dump_path.exists():
        print(f"Downloading Wikipedia XML dump (~22GB) ...")
        print(f"  URL: {DUMP_URL}")
        ret = subprocess.run(["wget", "-c", "-O", str(dump_path), DUMP_URL])
        if ret.returncode != 0:
            sys.exit("wget failed. Install wget or download manually.")
    else:
        print(f"Found existing dump: {dump_path}")

    # Extract with WikiExtractor
    if not extract_dir.exists():
        print("Extracting with WikiExtractor (this takes ~2 hours) ...")
        try:
            ret = subprocess.run(
                [
                    "python",
                    "-m",
                    "wikiextractor.WikiExtractor",
                    str(dump_path),
                    "--output",
                    str(extract_dir),
                    "--bytes",
                    "100M",
                    "--processes",
                    str(args.workers),
                    "--no-templates",
                    "--quiet",
                ],
            )
            if ret.returncode != 0:
                sys.exit("WikiExtractor failed. Install: pip install wikiextractor")
        except FileNotFoundError:
            sys.exit("wikiextractor not found. Install: pip install wikiextractor")
    else:
        print(f"Found existing extraction: {extract_dir}")

    # Read extracted articles
    _doc_re = re.compile(r"<doc[^>]*>(.*?)</doc>", re.DOTALL)
    articles = []
    for fpath in sorted(extract_dir.rglob("wiki_*")):
        content = fpath.read_text(encoding="utf-8", errors="replace")
        for m in _doc_re.finditer(content):
            text = clean_article(m.group(1))
            if len(text) > 100:
                articles.append(text)

    print(f"  {len(articles):,} articles extracted.")
    return articles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_split(articles, path):
    with open(path, "w", encoding="utf-8") as f:
        for art in articles:
            if art.strip():
                f.write(art + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  wrote {len(articles):,} articles → {path}  ({size_mb:.0f} MB)")


def main():
    args = parse_args()

    if args.source == "hf":
        articles = download_hf(args)
    else:
        articles = download_dump(args)

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(articles)
    n = len(articles)
    n_test = max(1, int(n * args.test_frac))
    n_valid = max(1, int(n * args.valid_frac))

    test_arts = articles[:n_test]
    valid_arts = articles[n_test : n_test + n_valid]
    train_arts = articles[n_test + n_valid :]

    print(
        f"\nSplit: {len(train_arts):,} train / "
        f"{len(valid_arts):,} valid / {len(test_arts):,} test"
    )
    print("Writing splits ...")
    write_split(train_arts, HERE / "train")
    write_split(valid_arts, HERE / "valid")
    write_split(test_arts, HERE / "test")

    print("\nDone.  Register the dataset in src/data/loader.py, then run:")
    print("  python run_all.py --data dataset/wikipedia/.train --words 1 \\")
    print("      --threads 4 --overlap 1 --cutoff 10000 --save_path wikipedia_train")


if __name__ == "__main__":
    main()
