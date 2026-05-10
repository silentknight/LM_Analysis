# LM Analysis

A toolkit for measuring complex system properties of language modeling datasets. Analyzes statistical and information-theoretic patterns that characterize natural language corpora.

## Analyses

- **Long-range Dependencies (LDDs)** — Mutual information between tokens at distance *d*, using standard or Grassberger entropy estimation
- **Pointwise Mutual Information (PMI)** — Token-pair PMI matrices across distances
- **Zipf's Law** — Rank-frequency distribution of the vocabulary
- **Heap's Law** — Vocabulary growth as a function of corpus size
- **Taylor's Law & Ebeling's Method** — Variance scaling of token frequencies over subsequences

## Project Structure

```
.
├── data.py                        # Corpus loading and tokenization
├── run_all.py                     # Main pipeline entry point
├── mutual_information.py          # LDD / MI computation (threaded)
├── pointwise_mutual_information.py# PMI computation (threaded)
├── plot_ldd.py                    # Plot LDD curves
├── recurrence.py                  # Recurrence analysis
├── speedup.pyx                    # Cython extension for fast joint RV computation
├── setup.py                       # Build script for Cython extension
├── main.c                         # C implementation of Heap's / Taylor's / Ebeling's laws
└── experiments/                   # Output directory (gitignored)
```

## Dependencies

- Python 3.x
- NumPy
- SciPy
- Cython
- simplejson
- matplotlib

## Installation

**1. Build the Cython extension:**

```bash
python setup.py build_ext --inplace
```

**2. Compile the C analysis binary:**

```bash
gcc -O2 -o main main.c -lm
```

**3. Create required output directories:**

```bash
mkdir -p experiments/{datasetInIDs,corpus,zipf,ldds,heaps,taylors,ebelings,recurrence,pmi}
```

## Usage

**Run the full analysis pipeline:**

```bash
python run_all.py --data dataset/wiki2/ \
                  --words 1 \
                  --cutoff 500 \
                  --threads 4 \
                  --mi_method grassberger \
                  --save_path wiki2
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Path to dataset directory | `dataset/wiki2/` |
| `--words` | Tokenize on words (`1`) or characters (`0`) | `0` |
| `--cutoff` | Maximum distance *d* for MI computation | — |
| `--mi_method` | `grassberger` or `standard` | `grassberger` |
| `--log_type` | Log base: `0`=e, `1`=2, `2`=10 | `1` |
| `--threads` | Number of parallel threads | `1` |
| `--overlap` | Allow overlapping windows (`1`/`0`) | `1` |
| `--save_path` | Output filename stem | `save_data.dat` |

**Plot LDD curves:**

```bash
python plot_ldd.py --path experiments/ldds/wiki2.dat --loglog 1
```

**Run Heap's / Taylor's / Ebeling's laws (C binary):**

```bash
./main wiki2
```

## License

Copyright (c) 2024 Abhijit. All Rights Reserved. See [LICENSE](LICENSE) for details.
