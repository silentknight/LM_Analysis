"""
Reorder test/valid Zipf arrays to match the train word frequency ranking.

Each dataset uses its own vocabulary ID space, so a word may have ID 5 in
the train set and ID 42 in the test set.  This script builds a correspondence
via the corpus dictionary (word string → ID) and re-sorts the test/valid
arrays so that word rank 1 in the output corresponds to the most-frequent
train word, rank 2 to the second-most-frequent train word, and so on.

Words present in train but absent from test/valid are skipped.  Words
present in test/valid but absent from train are dropped from the output.

Corpus dictionaries are loaded from <experiments_dir>/corpus/<name>_corpus.dat
(pickled data.Corpus objects from earlier runs).  The reordered Zipf arrays
are saved to <experiments_dir>/zipf/<name>_ordered.npz.

Usage example:
    python process_zipf.py \\
        --train  wiki2C_train \\
        --tests  wiki2MC_test1 wiki2MC_test2 \\
        --valids wiki2MC_valid1
"""

import argparse
import os
import pickle

import numpy as np


def load_zipf(path):
    with np.load(path) as arr:
        if "ids" in arr:
            return arr["ids"].copy(), arr["frequency"].copy()
        return arr["arr_0"].copy(), arr["arr_1"].copy()


def load_corpus(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def reorder_to_train(train_ids, train_corpus, test_ids, test_corpus):
    """Return reordered (ids, frequency) aligned to train word rank order."""
    # Map word string → current position in the test array
    test_vocab_size = len(test_corpus.dictionary.idx2word)
    test_pos = {
        test_corpus.dictionary.idx2word[wid]: pos
        for pos, wid in enumerate(test_ids)
        if wid < test_vocab_size
    }

    train_vocab_size = len(train_corpus.dictionary.idx2word)
    new_order = []
    absent = 0
    for wid in train_ids:
        if wid >= train_vocab_size:
            absent += 1
            continue
        word = train_corpus.dictionary.idx2word[wid]
        if word in test_pos:
            new_order.append(test_pos[word])
        else:
            absent += 1

    return np.array(new_order, dtype=np.intp), absent


def main():
    parser = argparse.ArgumentParser(
        description="Reorder test/valid Zipf arrays to match train word rank order"
    )
    parser.add_argument(
        "--experiments_dir",
        default="experiments",
        help="Experiments root directory (default: experiments)",
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Save-path name of the train split (e.g. wiki2C_train)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=[],
        help="Save-path names of test splits to reorder",
    )
    parser.add_argument(
        "--valids",
        nargs="+",
        default=[],
        help="Save-path names of valid splits to reorder",
    )
    args = parser.parse_args()

    zipf_dir = os.path.join(args.experiments_dir, "zipf")
    corpus_dir = os.path.join(args.experiments_dir, "corpus")

    # Load train
    train_ids, _ = load_zipf(os.path.join(zipf_dir, args.train + ".npz"))
    train_corpus = load_corpus(os.path.join(corpus_dir, args.train + "_corpus.dat"))

    for name in args.tests + args.valids:
        zipf_path = os.path.join(zipf_dir, name + ".npz")
        corpus_path = os.path.join(corpus_dir, name + "_corpus.dat")

        if not os.path.exists(zipf_path):
            print(f"Skipping {name}: zipf file not found ({zipf_path})")
            continue
        if not os.path.exists(corpus_path):
            print(f"Skipping {name}: corpus file not found ({corpus_path})")
            continue

        test_ids, test_freq = load_zipf(zipf_path)
        test_corpus = load_corpus(corpus_path)

        new_order, absent = reorder_to_train(
            train_ids, train_corpus, test_ids, test_corpus
        )
        out_ids = test_ids[new_order]
        out_freq = test_freq[new_order]

        out_path = os.path.join(zipf_dir, name + "_ordered.npz")
        np.savez_compressed(out_path, ids=out_ids, frequency=out_freq)
        print(
            f"{name}: reordered {len(new_order)} words, {absent} absent from test/valid"
        )


if __name__ == "__main__":
    main()
