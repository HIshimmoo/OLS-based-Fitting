import os
import re
import argparse
import logging

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import scipy._lib._util as _util

# Dynamically patch _lazywhere if missing (to support statsmodels-discrete)
if not hasattr(_util, '_lazywhere'):
    def _lazywhere(cond, func, x, y=None):
        cond_arr = np.asarray(cond)
        x_arr = np.asarray(x)
        y_arr = np.asarray(y) if y is not None else np.zeros_like(x_arr)
        return np.where(cond_arr, func(x_arr), y_arr)
    _util._lazywhere = _lazywhere

def parse_args():
    parser = argparse.ArgumentParser(
        description="OLS fitting with transform selection, single-x screening, and compact Excel output"
    )
    parser.add_argument(
        "--input", help="Input Excel file with xN and yN columns"
    )
    parser.add_argument(
        "--output", default="regression_results.xlsx",
        help="Output Excel file path"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for screening predictors"
    )
    parser.add_argument(
        "--transforms", default="linear,log,exp,power,reciprocal",
        help="Transforms to test (comma-separated)"
    )
    return parser.parse_args()

def apply_transform(X, y, transform):
    if transform == 'linear':
        return X, y
    if transform == 'log':
        return X.replace(0, np.nan).apply(np.log), y.replace(0, np.nan).apply(np.log)
    if transform == 'exp':
        return X, y.replace(0, np.nan).apply(np.log)
    if transform == 'power':
        return X.replace(0, np.nan).apply(np.log), y
    if transform == 'reciprocal':
        return X.replace(0, np.nan).rdiv(1), y
    raise ValueError(f"Unknown transform: {transform}")

def select_best_transform(X, y, transforms):
    best_type = None
    best_r2 = -np.inf
    best_model = None
    for t in transforms:
        try:
            X_t, y_t = apply_transform(X, y, t)
        except Exception as e:
            logging.warning(f"Skipping transform {t}: {e}")
            continue
        df = pd.concat([X_t, y_t.rename('y')], axis=1).dropna()
        if df.empty:
            continue
        X_mat = add_constant(df[X.columns], has_constant='add')
        model = OLS(df['y'], X_mat).fit()
        if model.rsquared > best_r2:
            best_r2 = model.rsquared
            best_type = t
            best_model = model
    if best_model is None:
        raise RuntimeError("No valid transform produced a model.")
    return best_type, best_model

def fit_all(Y, X, alpha, transforms):
    summary = {}
    screening_table = {}
    for resp in Y.columns:
        logging.info(f"Processing response: {resp}")
        y = Y[resp]
        # 1. Find best transform with all x
        init_t, init_model = select_best_transform(X, y, transforms)

        # 2. Single-variable screening for each x
        X_t, y_t = apply_transform(X, y, init_t)
        df = pd.concat([X_t, y_t.rename('y')], axis=1).dropna()
        eligible_x = []
        single_x_pvals = {}
        single_x_models = {}
        for xcol in X.columns:
            Xi = add_constant(df[[xcol]], has_constant='add')
            try:
                model_i = OLS(df['y'], Xi).fit()
                pval = model_i.pvalues.get(xcol, np.nan)
                model_type = f"y ~ {init_t}({xcol})"
            except Exception as e:
                logging.warning(f"Single-x OLS failed for {xcol}: {e}")
                pval = np.nan
                model_type = f"y ~ {init_t}({xcol}) [FAILED]"
            single_x_pvals[xcol] = pval
            single_x_models[xcol] = model_type
            if not np.isnan(pval) and pval < alpha:
                eligible_x.append(xcol)
        screening_table[resp] = {x: single_x_pvals[x] for x in X.columns}

        # 3. Final fit with all eligible x (or fallback to intercept)
        if eligible_x:
            X_mat_final = add_constant(df[eligible_x], has_constant='add')
            final_model = OLS(df['y'], X_mat_final).fit()
            final_coefs = final_model.params.to_dict()
            final_p = final_model.pvalues.to_dict()
            final_r2 = final_model.rsquared
            final_n = len(final_model.params) - (1 if 'const' in final_model.params else 0)
            model_str = f"y ~ {init_t}({', '.join(eligible_x)})"
        else:
            df0 = df[['y']]
            X0 = add_constant(pd.DataFrame(index=df0.index), has_constant='add')
            final_model = OLS(df0['y'], X0).fit()
            final_coefs = final_model.params.to_dict()
            final_p = final_model.pvalues.to_dict()
            final_r2 = final_model.rsquared
            final_n = 0
            model_str = "y ~ 1 (Intercept-only)"

        summary[resp] = {
            'transform': init_t,
            'final_model_str': model_str,
            'final_r2': final_r2,
            'final_coefs': final_coefs,
            'final_p': final_p,
            'final_n': final_n
        }
    return summary, screening_table

def collect_all_single_x_p(Y, X, transforms):
    # Returns: dict of {transform: DataFrame of p-values [rows=y, cols=x]}
    results = {t: pd.DataFrame(index=Y.columns, columns=X.columns) for t in transforms}
    for resp in Y.columns:
        y = Y[resp]
        for t in transforms:
            for x in X.columns:
                try:
                    X1, y1 = apply_transform(X[[x]], y, t)
                except Exception:
                    results[t].loc[resp, x] = np.nan
                    continue
                df1 = pd.concat([X1, y1.rename('y')], axis=1).dropna()
                if df1.empty:
                    results[t].loc[resp, x] = np.nan
                    continue
                X1_mat = add_constant(df1[[x]], has_constant='add')
                try:
                    model_i = OLS(df1['y'], X1_mat).fit()
                    pval = model_i.pvalues.get(x, np.nan)
                except Exception:
                    pval = np.nan
                results[t].loc[resp, x] = pval
    return results

def main():
    args = parse_args()
    transforms = [t.strip() for t in args.transforms.split(',')]
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input_file = args.input or next(
        (f for f in os.listdir('.') if f.lower().endswith(('.xls', '.xlsx'))), None
    )
    if not input_file:
        raise FileNotFoundError('No Excel file found in current directory.')
    logging.info(f"Reading input file: {input_file}")
    df = pd.read_excel(input_file)

    var_pattern = re.compile(r'^[xy]\d+$', re.IGNORECASE)
    vars_found = [c for c in df.columns if var_pattern.match(c)]
    if not vars_found:
        raise ValueError('No columns matching xN or yN found.')
    X = df[[c for c in vars_found if c.lower().startswith('x')]]
    Y = df[[c for c in vars_found if c.lower().startswith('y')]]

    summary, screening_table = fit_all(Y, X, args.alpha, transforms)

    def build_formula(coefs, model_vars):
        # model_vars: list of variable names (excluding 'const')
        terms = []
        for var in model_vars:
            coef = coefs.get(var, np.nan)
            if pd.notnull(coef):
                terms.append(f"{coef:+.4g}*{var}")
        intercept = coefs.get('const', 0)
        rhs = f"{intercept:.4g} " + " ".join(terms)
        return rhs

    def full_x_best_r2_fit(Y, X, transforms):
        results = []
        for resp in Y.columns:
            y = Y[resp]
            best_t, best_model = select_best_transform(X, y, transforms)
            coefs = best_model.params.to_dict()
            pvals = best_model.pvalues.to_dict()
            r2 = best_model.rsquared
            model_vars = [v for v in X.columns if v in coefs]
            formula = build_formula(coefs, model_vars)
            results.append({
                'Response': resp,
                'Model (best R2, all x)': f"y ~ {best_t}({', '.join(model_vars)})",
                'Formula (best R2)': f"{resp} = {formula}",
                'R2 (best R2)': r2,
            })
        return pd.DataFrame(results)

    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        summary_rows = []
        for resp, info in summary.items():
            model_vars = [v for v in info['final_coefs'] if v != 'const']
            formula = build_formula(info['final_coefs'], model_vars)
            row = {
                'Response': resp,
                'Model': info['final_model_str'],
                'Transform': info['transform'],
                'Formula': f"{resp} = {formula}",
                'Final R2': info['final_r2'],
                'Final #Predictors': info['final_n']
            }
            # Add each variable's coefficient and p-value (as extra columns)
            for v in info['final_coefs']:
                row[f"Coef({v})"] = info['final_coefs'][v]
                row[f"P({v})"] = info['final_p'].get(v, np.nan)
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)

        # Main summary sheet (screened model)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Best-R2 (full-x) fit for each y, appended below summary
        full_r2_df = full_x_best_r2_fit(Y, X, transforms)
        # Write it below the summary on same sheet:
        startrow = len(summary_df) + 2
        full_r2_df.to_excel(writer, sheet_name='Summary', index=False, startrow=startrow)

        # Screening P table (still useful to have)
        screen_out = []
        for resp, pvals in screening_table.items():
            row = {'Response': resp}
            row.update({x: ("{:.4f}".format(p) if pd.notnull(p) else "NaN") for x, p in pvals.items()})
            screen_out.append(row)
        pd.DataFrame(screen_out).to_excel(writer, sheet_name='Screening P', index=False)

    logging.info(f"Analysis complete. Results saved to '{args.output}'.")


if __name__ == '__main__':
    main()
