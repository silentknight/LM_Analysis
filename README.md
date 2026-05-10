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
├── data.py          # Corpus loading and tokenization
├── run_all.py       # Main pipeline entry point
├── mi.py            # LDD / MI computation (threaded)
├── pmi.py           # PMI computation (threaded)
├── recurrence.py    # Recurrence analysis
├── speedup.pyx      # Cython extension for fast joint RV computation
├── setup.py         # Build script for Cython extension
├── main.c           # C implementation of Heap's / Taylor's / Ebeling's laws
├── validate.py      # Baseline validation tool
├── experiments/     # Output directory (gitignored)
└── baseline/        # Saved baselines for validation (gitignored)
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
gcc -O2 -o main main.c -lm
```

## Usage

**Run the full analysis pipeline:**

```bash
python -u run_all.py --data dataset/ptb/.full \
                     --words 1 \
                     --cutoff 1000 \
                     --threads 4 \
                     --log_type 0 \
                     --mi_method grassberger \
                     --save_path ptb_full
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset registry key (see `data.py`) | required |
| `--words` | Tokenize on words (`1`) or characters (`0`) | `0` |
| `--cutoff` | Maximum distance *d* for MI / PMI computation | `1000` |
| `--mi_method` | `grassberger` or `standard` | `grassberger` |
| `--log_type` | Log base: `0`=e, `1`=2, `2`=10 | `1` |
| `--threads` | Number of parallel threads | `1` |
| `--overlap` | Allow overlapping windows (`1`/`0`) | `1` |
| `--save_path` | Output filename stem | `save_data` |
| `--clear` | Clear existing LDD file before running (`1`/`0`) | `0` |
| `--processes` | Analyses to run (space-separated subset of `mi pmi heaps recurrence`) | all |

**Run a subset of analyses:**

```bash
# MI only
python run_all.py --data dataset/ptb/.full --save_path ptb_full --processes mi

# Heaps / Taylor's / Ebeling's only (auto-compiles main.c if needed)
python run_all.py --data dataset/ptb/.full --save_path ptb_full --processes heaps
```

## Output Files

All outputs are written under `experiments/`:

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
    --run_cmd "python -u run_all.py --data dataset/ptb/.full --words 1 \
               --threads 10 --log_type 0 --mi_method grassberger \
               --overlap 1 --cutoff 100 --clear 1 --save_path ptb_full \
               --processes mi heaps recurrence"
```

After making code changes, re-run and verify outputs stay within tolerance:

```bash
python validate.py check --save_path ptb_full
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
