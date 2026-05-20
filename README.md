# LM Analysis

A toolkit for measuring complex system properties of language modeling datasets. Analyzes statistical and information-theoretic patterns that characterize natural language corpora.

## Analyses

- **Long-range Dependencies (LDDs)** — Mutual information between tokens at distance *d*, using standard or Grassberger entropy estimation
- **Pointwise Mutual Information (PMI)** — Token-pair PMI matrices across distances
- **Zipf's Law** — Rank-frequency distribution of the vocabulary
- **Heap's Law** — Vocabulary growth as a function of corpus size
- **Taylor's Law & Ebeling's Method** — Variance scaling of token frequencies over subsequences
- **Recurrence** — Inter-occurrence gap distributions for each vocabulary token

## Project Structure

```
.
├── run_all.py           # Main pipeline entry point
├── validate.py          # Baseline validation tool
├── setup.py             # Build script for the Cython extension
│
└── src/
    ├── data/            # Corpus loading and tokenization
    │   └── loader.py
    │
    ├── analysis/        # Computation modules
    │   ├── mi.py        # LDD / MI computation (threaded)
    │   ├── pmi.py       # PMI computation (threaded)
    │   ├── recurrence.py
    │   ├── process_zipf.py  # Reorder Zipf arrays to match train word rank
    │   ├── speedup.pyx  # Cython extension for fast joint RV computation
    │   └── scaling_laws.c       # C implementation of Heap's / Taylor's / Ebeling's laws
    │
    ├── utils/           # Shared utilities
    │   ├── loaders.py   # Loaders for experiment output files (npz, csv)
    │   └── math_utils.py  # Power law fitting and evaluation
    │
    └── plots/           # Visualisation and error estimation
        ├── plot_all.py  # Plot all analysis types
        ├── plot_utils.py
        ├── datasets.py  # Dataset group configuration
        └── estimate_zipf_error.py

experiments/             # Output directory (gitignored)
baseline/                # Saved baselines for validation (gitignored)
```

## Dependencies

- Python 3.x
- NumPy
- SciPy
- Cython
- matplotlib

## Installation

**1. Build the Cython extension:**

```bash
python setup.py build_ext --inplace
```

**2. Compile the C analysis binary** (done automatically by `run_all.py` if not present):

```bash
gcc -O2 -o src/analysis/scaling_laws src/analysis/scaling_laws.c -lm
```

## Usage

**Run the full analysis pipeline:**

```bash
python -u run_all.py --data dataset/text/english/ptb/.full \
                     --save_path ptb_full \
                     --tokenizer word \
                     --cutoff 1000 \
                     --threads 4 \
                     --log_base e \
                     --mi_method grassberger
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Registry key for the corpus (see `src/data/loader.py`) | **required** |
| `--save_path` | Stem for all output filenames | **required** |
| `--tokenizer` | Tokenizer: `word`, `char`, … (extensible) | `word` |
| `--cutoff` | Maximum distance *d* for MI / PMI | `1000` |
| `--mi_method` | Entropy estimator: `grassberger` or `standard` | `grassberger` |
| `--log_base` | Log base for MI / PMI values: `e`, `2`, or `10` | `2` |
| `--threads` | Parallel workers | `1` |
| `--no_overlap` | Disallow overlapping token windows (flag) | off |
| `--clear` | Delete existing LDD file before MI (flag) | off |
| `--processes` | Analyses to run: `mi pmi heaps taylors ebelings recurrence` (`heaps`/`taylors`/`ebelings` share one C binary) | `mi heaps taylors ebelings recurrence` |

**Run a subset of analyses:**

```bash
# MI only
python run_all.py --data dataset/text/english/ptb/.full --save_path ptb_full --processes mi

# Heaps / Taylor's / Ebeling's only (auto-compiles scaling_laws.c if needed)
python run_all.py --data dataset/text/english/ptb/.full --save_path ptb_full --processes heaps taylors ebelings
```

## Output Files

All outputs are written under `experiments/<data_type>/<language>/<tokenizer>/`
(e.g. `experiments/text/english/word/`). Override with `--experiments_dir`.

| Directory | File | Format |
|---|---|---|
| `datasetInIDs/` | `<save_path>.csv` | One token ID per line |
| `zipf/` | `<save_path>.npz` | Arrays `ids`, `frequency` |
| `ldds/` | `<save_path>.csv` | CSV with columns `d,mi,Hx,Hy,Hxy` |
| `heaps/` | `<save_path>_heaps.csv` | Comma-separated cumulative vocab sizes |
| `taylors/` | `<save_path>_taylors.csv` | Alternating window-size / mean / SD rows |
| `ebelings/` | `<save_path>_ebelings.csv` | `window:total_variance` entries |
| `recurrence/` | `<save_path>.npz` | Per-word arrays of shape `(2, N)`: gap lengths and counts |
| `pmi/<save_path>/` | `marginals/`, `Ni_XY/`, `pmi/` | Per-distance compressed sparse matrices |

## Validation

Save a named baseline from the current outputs:

```bash
python validate.py save --save_path ptb_full \
    --experiments_dir experiments/text/english/word \
    --run_cmd "python -u run_all.py --data dataset/text/english/ptb/.full --tokenizer word \
               --threads 10 --log_base e --mi_method grassberger \
               --cutoff 100 --clear --save_path ptb_full \
               --processes mi heaps taylors ebelings recurrence"
```

After making code changes, re-run and verify outputs stay within tolerance:

```bash
python validate.py check --save_path ptb_full --experiments_dir experiments/text/english/word
```

Compare existing outputs without re-running:

```bash
python validate.py check --save_path ptb_full --skip_run
```

Tolerances applied per output:

| Output | Tolerance |
|---|---|
| `datasetInIDs`, `zipf`, `heaps` | Exact match |
| `ldds` | Absolute ≤ 1e-10 |
| `taylors` | Absolute ≤ 1e-6 or relative ≤ 1% |
| `ebelings` | Absolute ≤ 1e-4 or relative ≤ 1% |
| `recurrence` | Reconstructed token count must match |

## License

Copyright (c) 2024 Abhijit Mahalunkar. All Rights Reserved. See [LICENSE](LICENSE) for details.
