
# loo_cv_regression_singleX.py
import os
import re
import argparse
import logging

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import LeaveOneOut
import scipy._lib._util as _util

# ---------------------------------------------------------------------
# Patch statsmodels’ lazywhere if needed (env compatibility safeguard)
# ---------------------------------------------------------------------
def _patch_lazywhere():
    if not hasattr(_util, '_lazywhere'):
        def _lazywhere(cond, func, x, y=None):
            cond_arr = np.asarray(cond)
            x_arr = np.asarray(x)
            y_arr = np.asarray(y) if y is not None else np.zeros_like(x_arr)
            return np.where(cond_arr, func(x_arr), y_arr)
        _util._lazywhere = _lazywhere
_patch_lazywhere()

# ---------------------------------------------------------------------
# Transform configuration (updated)
# ---------------------------------------------------------------------
TRANSFORMS_DEFAULT = ["linear", "log", "exp", "reciprocal"]

TRANSFORM_DESC = {
    "linear":     "X raw; Y raw",
    "log":        "X log10(x); Y raw",
    "exp":        "X raw; Y log(y)",
    "reciprocal": "X 1/x; Y raw",
}

def expr(var, transform):
    if transform == "log":
        return f"log10({var})"
    if transform == "reciprocal":
        return f"1/({var})"
    return var

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LOO-CV screening (single-X only), Q2 heatmaps, and per-transform single-X models"
    )
    parser.add_argument("--input", help="Input Excel file with xN and yN columns (case-insensitive)")
    parser.add_argument("--output", default="regression_results_singleX.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--transforms", default=",".join(TRANSFORMS_DEFAULT),
                        help="Transforms to test (comma-separated, from: linear,log,exp,reciprocal)")
    return parser.parse_args()

# ---------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------
def _log10_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.astype(float).copy()
    out[~(out > 0)] = np.nan
    return np.log10(out)

def _ln_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s[~(s > 0)] = np.nan
    return np.log(s)

def _reciprocal_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.astype(float).copy()
    out[out == 0] = np.nan
    return 1.0 / out

def apply_transform(X: pd.DataFrame, y: pd.Series, transform: str):
    if transform == "linear":
        return X.copy(), y.copy()
    if transform == "log":          # X -> log10(X); Y raw
        return _log10_safe(X), y.copy()
    if transform == "exp":          # Y -> ln(Y); X raw
        return X.copy(), _ln_safe(y)
    if transform == "reciprocal":   # X -> 1/X; Y raw
        return _reciprocal_safe(X), y.copy()
    raise ValueError(f"Unknown transform: {transform}")

# ---------------------------------------------------------------------
# LOO–Q2 for a single predictor
# ---------------------------------------------------------------------
def loo_q2_singlex(x: pd.Series, y: pd.Series, transform: str) -> float:
    X_df = x.to_frame()
    X_t, y_t = apply_transform(X_df, y, transform)
    df = pd.concat([X_t, y_t.rename("y")], axis=1).dropna()
    if df.shape[0] < 3:
        return np.nan

    loo = LeaveOneOut()
    press = 0.0
    y_vals = df["y"].values
    y_mean = np.nanmean(y_vals)

    for train_idx, test_idx in loo.split(df):
        train = df.iloc[train_idx]
        test  = df.iloc[test_idx]
        Xi = add_constant(train[[x.name]], has_constant="add")
        model = OLS(train["y"], Xi).fit()
        X_test = add_constant(test[[x.name]], has_constant="add")
        y_pred = model.predict(X_test).values[0]
        press += (test["y"].values[0] - y_pred) ** 2

    ss_tot = np.nansum((y_vals - y_mean) ** 2)
    return 1 - press / ss_tot if ss_tot > 0 else np.nan

# ---------------------------------------------------------------------
# Collect per-transform heatmaps (single X only)
# ---------------------------------------------------------------------
def collect_all_loo_q2(Y: pd.DataFrame, X: pd.DataFrame, transforms):
    results = {t: pd.DataFrame(index=Y.columns, columns=X.columns, dtype=float) for t in transforms}
    for resp in Y.columns:
        logging.info(f"Screening {resp} by LOO-Q2 (single X)...")
        y = Y[resp]
        for t in transforms:
            for xcol in X.columns:
                try:
                    q2 = loo_q2_singlex(X[xcol], y, t)
                except Exception:
                    q2 = np.nan
                results[t].at[resp, xcol] = q2
    return results

# ---------------------------------------------------------------------
# Build a best-single-Q2 heatmap across transforms
# ---------------------------------------------------------------------
def best_single_q2_heatmap(screening_q2_dict):
    transforms = list(screening_q2_dict.keys())
    any_df = screening_q2_dict[transforms[0]]
    responses = any_df.index
    predictors = any_df.columns
    best = pd.DataFrame(index=responses, columns=predictors, dtype=float)
    for r in responses:
        for c in predictors:
            vals = [screening_q2_dict[t].at[r, c] for t in transforms]
            vals = [v for v in vals if pd.notna(v)]
            best.at[r, c] = max(vals) if len(vals) > 0 else float('nan')
    return best

# ---------------------------------------------------------------------
# Select best model per response (single predictor only)
# ---------------------------------------------------------------------
def select_best_model(Y: pd.DataFrame, X: pd.DataFrame, transforms):
    rows = []
    for resp in Y.columns:
        y = Y[resp]
        best = {"q2": -np.inf, "t": None, "x": None}

        for t in transforms:
            for xcol in X.columns:
                q2 = loo_q2_singlex(X[xcol], y, t)
                if pd.notnull(q2) and q2 > best["q2"]:
                    best = {"q2": q2, "t": t, "x": xcol}

        if best["x"] is not None:
            X_t, y_t = apply_transform(X[[best["x"]]], y, best["t"])
            df_best = pd.concat([X_t, y_t.rename("y")], axis=1).dropna()
            Xi = add_constant(df_best[[best["x"]]], has_constant="add")
            model = OLS(df_best["y"], Xi).fit()
            coefs = model.params
            r2    = model.rsquared
            intercept = coefs.get("const", 0.0)
            slope     = coefs.get(best["x"], np.nan)
            formula = f"{resp} = {intercept:.4g} {slope:+.4g}*{expr(best['x'], best['t'])}"
            rows.append({
                "Response": resp,
                "Transformation": best["t"],
                "Best Subset": best["x"],
                "Q2": best["q2"],
                "R2": r2,
                "Formula": formula,
                "Notes": TRANSFORM_DESC.get(best["t"], "")
            })
        else:
            rows.append({
                "Response": resp,
                "Transformation": None,
                "Best Subset": "",
                "Q2": np.nan,
                "R2": np.nan,
                "Formula": "No valid single-X model",
                "Notes": ""
            })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Build per-transform single-XY model tables
# ---------------------------------------------------------------------
def build_single_xy_tables(Y: pd.DataFrame, X: pd.DataFrame, transforms):
    tables = {}
    for t in transforms:
        rows = []
        for resp in Y.columns:
            y = Y[resp]
            for xcol in X.columns:
                q2 = loo_q2_singlex(X[xcol], y, t)
                if pd.isna(q2):
                    rows.append({
                        "Response": resp, "Predictor": xcol, "Q2": np.nan,
                        "R2": np.nan, "Formula": "No valid model"
                    })
                    continue
                X_t, y_t = apply_transform(X[[xcol]], y, t)
                df_pair = pd.concat([X_t, y_t.rename("y")], axis=1).dropna()
                Xi = add_constant(df_pair[[xcol]], has_constant="add")
                model = OLS(df_pair["y"], Xi).fit()
                coefs = model.params
                r2 = model.rsquared
                intercept = coefs.get("const", 0.0)
                slope     = coefs.get(xcol, np.nan)
                formula = f"{resp} = {intercept:.4g} {slope:+.4g}*{expr(xcol, t)}"
                rows.append({
                    "Response": resp,
                    "Predictor": xcol,
                    "Q2": q2,
                    "R2": r2,
                    "Formula": formula
                })
        tables[t] = pd.DataFrame(rows)
    return tables

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    transforms = [t.strip() for t in args.transforms.split(',') if t.strip()]
    for t in transforms:
        if t not in TRANSFORMS_DEFAULT:
            raise ValueError(f"Unsupported transform: {t}. Choose from {TRANSFORMS_DEFAULT}" )

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1) Read data
    input_file = args.input or next((f for f in os.listdir('.') if f.lower().endswith(('.xls','.xlsx'))), None)
    if not input_file:
        raise FileNotFoundError('No Excel file found.')
    df = pd.read_excel(input_file)

    var_pattern = re.compile(r'^[xy]\d+$', re.IGNORECASE)
    vars_found = [c for c in df.columns if var_pattern.match(c)]
    if not vars_found:
        raise ValueError("No columns like x1, x2, ..., y1, y2, ... found.")

    X = df[[c for c in vars_found if c.lower().startswith('x')]]
    Y = df[[c for c in vars_found if c.lower().startswith('y')]]

    # 2) Compute single-X Q2 heatmaps
    screening_q2 = collect_all_loo_q2(Y, X, transforms)
    # 2b) Best-single Q2 heatmap
    best_single_q2 = best_single_q2_heatmap(screening_q2)

    # 3) Best single-X model per response
    best_models = select_best_model(Y, X, transforms)

    # 4) Detailed single-XY tables for each transform
    per_transform_tables = build_single_xy_tables(Y, X, transforms)

    # 5) Write to Excel
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        # Best single-Q2
        best_single_q2.to_excel(writer, sheet_name='Best_Single_Q2')

        # Heatmaps per family
        for t, df_q2 in screening_q2.items():
            df_q2.to_excel(writer, sheet_name=f"Heatmap_Q2_{t}")

        # Best model table
        cols_order = ["Response", "Transformation", "Best Subset", "Q2", "R2", "Formula", "Notes"]
        best_models[cols_order].to_excel(writer, sheet_name='Best_Model', index=False)

        # Detailed Single-XY per transform
        for t, tbl in per_transform_tables.items():
            tbl.to_excel(writer, sheet_name=f"Single_XY_{t}", index=False)

    logging.info(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
