# OLS-based Fitting

This script fits ordinary least squares models for variables found in an Excel worksheet. It automatically searches for columns named like `x1`, `y1`, etc., tries several transformations, removes insignificant predictors and outputs the results to a new Excel file.

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
