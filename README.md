# OLS-based Fitting

This script fits ordinary least squares models for variables found in an Excel worksheet. It automatically searches for columns named like `x1`, `y1`, etc., tries several transformations, removes insignificant predictors and outputs the results to a new Excel file.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Note:** `scipy>=1.11` is required for compatibility with statsmodels.

## Usage

```
python Fit.py --input data.xlsx --output results.xlsx --alpha 0.05
```

Options:

- `--input` – path to the Excel file. If omitted, the first `.xlsx`/`.xls` file in the current directory is used.
- `--output` – path of the results workbook (default `regression_results.xlsx`).
- `--alpha` – significance threshold when removing predictors (default `0.05`).
- `--transforms` – comma separated list of transforms to try (`linear,log,exp,power` by default).

The output workbook contains summary statistics as well as coefficient and p‑value sheets for each response variable.

## Workflow

1. **Discover and Load Data**
   - Searches the working directory for the first Excel file if `--input` is not provided.
   - Reads that file into a DataFrame.
2. **Detect Variables by Name**
   - Columns named like `xN` are predictors and `yN` are responses.
3. **Initial Transform Selection**
   - Each response is fit using four transformation options (linear, log–log, semi-log, power).
   - The transform with the highest R² is chosen.
4. **Iterative Refinement**
   - Drop the predictor with the worst p-value above `--alpha`.
   - Refit until all remaining terms are significant.
5. **Export Results**
   - Summaries, coefficients and p-values are written to the output Excel workbook.


## Troubleshooting

If you encounter an error like:

```
ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'
```

make sure you have upgraded **SciPy** to at least version 1.11:

```bash
pip install --upgrade "scipy>=1.11"
```

