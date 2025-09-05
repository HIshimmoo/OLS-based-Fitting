# Leave-One-Out Single-X Regression

This repository contains `loo_cv_regression_singleX.py`, a utility for
screening single-variable ordinary least squares (OLS) models using
leave-one-out cross-validation (LOO-CV).

---

## 1. Introduction

- **Goal:** For each response column `yN`, evaluate every predictor `xN`
  under several transformations and compute LOO-Q².
- **Result:** The best single-X model for each response along with
  diagnostic heatmaps and detailed tables.

---

## 2. Prerequisites

- **Python 3.7+**
- **Packages:** `pandas`, `numpy`, `statsmodels`, `scikit-learn`,
  `openpyxl`

Install via:

```bash
pip install pandas numpy statsmodels scikit-learn openpyxl
```

---

## 3. Usage

```bash
python loo_cv_regression_singleX.py --input input.xlsx --output results.xlsx --transforms linear,log,exp,reciprocal
```

- `--input`: Excel file containing columns named `x1, x2, ...` and
  `y1, y2, ...`. If omitted, the script uses the first `.xls` or `.xlsx`
  file in the working directory.
- `--output`: Name of the generated Excel report (default:
  `regression_results_singleX.xlsx`).
- `--transforms`: Comma-separated list of transforms to test. Supported
  options: `linear`, `log`, `exp`, `reciprocal`.

---

## 4. Methodology

For each pair `(y_j, x_i)` the script:

1. Applies the specified transformation family.
2. Fits an OLS model to the remaining `n-1` samples for every holdout.
3. Computes **LOO-Q²** using `Q² = 1 - PRESS / SST`.
4. Repeats for all predictors, responses and transforms.

Available transforms:

| Transform    | X            | Y       |
|--------------|--------------|---------|
| `linear`     | raw          | raw     |
| `log`        | `log10(x)`   | raw     |
| `exp`        | raw          | `ln(y)` |
| `reciprocal` | `1/x`        | raw     |

*This script evaluates single predictors only; multivariate subset
selection is not performed.*

---

## 5. Output

An Excel workbook is produced with these sheets:

- `Best_Single_Q2`: best Q² per `(y, x)` across transformations.
- `Heatmap_Q2_<transform>`: LOO-Q² heatmaps for each transform family.
- `Best_Model`: summary of the top single-X model for each response.
- `Single_XY_<transform>`: detailed coefficients and statistics for each
  `(y, x)` pair under that transform.

---

## 6. Customization

- Provide a subset of transforms via `--transforms`.
- Examine heatmaps to gauge predictive power; higher Q² indicates better
  out-of-sample performance.

