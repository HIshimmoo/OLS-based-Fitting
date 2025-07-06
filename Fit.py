import os
import re
import argparse
import logging

import scipy
from packaging import version
if version.parse(scipy.__version__) < version.parse("1.11"):
    raise ImportError(f"SciPy >= 1.11 required, found {scipy.__version__}")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def parse_args():
    parser = argparse.ArgumentParser(description="OLS fitting with reselection")
    parser.add_argument("--input", help="Input Excel file path")
    parser.add_argument("--output", default="regression_results.xlsx", help="Output Excel file")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for predictor removal")
    parser.add_argument("--transforms", default="linear,log,exp,power",
                        help="Comma-separated list of transforms to try")
    return parser.parse_args()

# === Helper functions for model selection & refinement ===
def select_best_transform(X, y, transforms=('linear','log','exp','power')):
    """
    Fit multiple transform-based OLS models, return the one with highest R².
    """
    best_type, best_r2, best_model = None, -np.inf, None
    for t in transforms:
        logging.info("Trying %s transform", t)
        df = X.copy()
        df['y'] = y
        if t == 'linear':
            formula = 'y ~ ' + ' + '.join(X.columns)
        elif t == 'log':
            # log-log model: log(y) ~ log(x_i)
            df = df.apply(lambda c: np.log(c.replace(0, np.nan)))
            df['y'] = np.log(y.replace(0, np.nan))
            formula = 'y ~ ' + ' + '.join(X.columns)
        elif t == 'exp':
            # semi-log: log(y) ~ x_i
            df['y'] = np.log(y.replace(0, np.nan))
            formula = 'y ~ ' + ' + '.join(X.columns)
        elif t == 'power':
            # y ~ log(x_i)
            df = df.apply(lambda c: np.log(c.replace(0, np.nan)))
            formula = 'y ~ ' + ' + '.join(X.columns)
        else:
            continue
        df = df.dropna()
        if df.empty:
            continue
        model = smf.ols(formula, data=df).fit()
        if model.rsquared > best_r2:
            best_type, best_r2, best_model = t, model.rsquared, model
    return best_type, best_model


def refine_with_reselection(X, y, alpha=0.05, transforms=('linear','log','exp','power')):
    """
    Iteratively select best transform, drop highest-p predictor until all p <= alpha.
    Returns final transform type and fitted model.
    """
    predictors = list(X.columns)
    final_type, final_model = None, None

    while predictors:
        logging.info("Fitting with predictors: %s", ', '.join(predictors))
        ttype, model = select_best_transform(X[predictors], y, transforms)
        pvals = model.pvalues.drop('Intercept', errors='ignore').dropna()
        if pvals.empty or pvals.max() <= alpha:
            final_type, final_model = ttype, model
            break
        # drop worst predictor and retry
        worst = pvals.idxmax()
        logging.info("Dropping %s (p=%.3f)", worst, pvals[worst])
        predictors.remove(worst)

    # if no predictors left, fit intercept-only
    if final_model is None:
        df = pd.DataFrame({'y': y})
        final_model = smf.ols('y ~ 1', data=df).fit()
        final_type = 'intercept-only'

    return final_type, final_model


def fit_all_resel(Y, X, alpha=0.05, transforms=('linear','log','exp','power')):
    """
    Fit for each response column in Y with predictors X.
    Returns a dict of summary info.
    """
    results = {}
    for resp in Y.columns:
        logging.info("Processing response %s", resp)
        y = Y[resp]
        init_type, init_model = select_best_transform(X, y, transforms)
        final_type, final_model = refine_with_reselection(X, y, alpha, transforms)
        # collate summary
        coef = final_model.params.to_dict()
        results[resp] = {
            'initial_transform': init_type,
            'initial_r2': init_model.rsquared if init_model else np.nan,
            'final_transform': final_type,
            'final_r2': final_model.rsquared,
            'coefficients': coef,
            'pvalues': final_model.pvalues.to_dict(),
            'n_predictors': len(final_model.params.drop('Intercept', errors='ignore'))
        }
    return results


# === Main execution ===
def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    input_file = args.input
    if input_file is None:
        files = [f for f in os.listdir('.') if f.lower().endswith(('.xlsx','.xls'))]
        if not files:
            raise FileNotFoundError("No Excel file found in directory.")
        input_file = files[0]
        logging.info("Using input file: %s", input_file)

    # read data
    df = pd.read_excel(input_file)

    # detect variables: names starting with x or y followed by digits
    var_pattern = re.compile(r'^[xy]\d+$', re.IGNORECASE)
    vars_found = [c for c in df.columns if var_pattern.match(c)]
    if not vars_found:
        raise ValueError("No variables matching pattern xN or yN found.")

    # separate X and Y
    X = df[[c for c in vars_found if c.lower().startswith('x')]]
    Y = df[[c for c in vars_found if c.lower().startswith('y')]]

    # fit models
    summary = fit_all_resel(Y, X, alpha=args.alpha, transforms=args.transforms)

    # prepare output Excel
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        # summary sheet
        summary_rows = []
        for resp, info in summary.items():
            row = {
                'response': resp,
                'initial_transform': info['initial_transform'],
                'initial_r2': info['initial_r2'],
                'final_transform': info['final_transform'],
                'final_r2': info['final_r2'],
                'n_predictors': info['n_predictors']
            }
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

        # coefficients and p-values in separate sheets
        coef_df = pd.DataFrame(
            {resp: info['coefficients'] for resp, info in summary.items()}
        ).T
        coef_df.to_excel(writer, sheet_name='Coefficients')

        pval_df = pd.DataFrame(
            {resp: info['pvalues'] for resp, info in summary.items()}
        ).T
        pval_df.to_excel(writer, sheet_name='P-values')

    logging.info("Results written to %s", args.output)

if __name__ == '__main__':
    args = parse_args()
    args.transforms = [t.strip() for t in args.transforms.split(',') if t.strip()]
    main(args)
