import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Define the transform selection and refinement functions
def select_best_model(X, y, model_types=['linear','log','exp','power']):
    best_type, best_r2, best_model = None, -np.inf, None
    for m in model_types:
        df = X.copy()
        df['y'] = y
        if m == 'linear':
            formula = 'y ~ ' + ' + '.join(X.columns)
        elif m == 'log':
            df = df.apply(lambda c: np.log(c.replace(0, np.nan)))
            df['y'] = np.log(y.replace(0, np.nan))
            formula = 'y ~ ' + ' + '.join(X.columns)
        elif m == 'exp':
            df['y'] = np.log(y.replace(0, np.nan))
            formula = 'y ~ ' + ' + '.join(X.columns)
        elif m == 'power':
            df = df.apply(lambda c: np.log(c.replace(0, np.nan)))
            formula = 'y ~ ' + ' + '.join(X.columns)
        else:
            continue
        df = df.dropna()
        model = smf.ols(formula, data=df).fit()
        if model.rsquared > best_r2:
            best_type, best_r2, best_model = m, model.rsquared, model
    return best_type, best_model

def refine_with_transform_selection(X, y, alpha=0.05, model_types=['linear','log','exp','power']):
    remaining = list(X.columns)
    final_type, final_model = None, None

    while True:
        mtype, model = select_best_model(X[remaining], y, model_types)
        pvals = model.pvalues.drop('Intercept', errors='ignore')
        if pvals.empty or pvals.max() <= alpha:
            final_type, final_model = mtype, model
            break
        worst = pvals.idxmax()
        remaining.remove(worst)

    return final_type, final_model

def fit_all_reselect(Y, X, alpha=0.05):
    summary = {}
    for col in Y.columns:
        y = Y[col]
        init_type, init_model = select_best_model(X, y)
        final_type, final_model = refine_with_transform_selection(X, y, alpha)
        summary[col] = {
            'initial_model_type': init_type,
            'initial_r2': init_model.rsquared,
            'final_model_type': final_type,
            'final_r2': final_model.rsquared,
            'final_params': final_model.params
        }
    return summary

# Create data from the user's six-sample table
samples = ['S1','S2','S3','S4','S5','S6']
X = pd.DataFrame({
    'x1': [0.194, 0.265, 0.271, 0.351, 0.28033, 0.232],
    'x2': [3.587, 4.25867, 5.76067, 8.342, 9.36997, 19.52833]
}, index=samples)

Y = pd.DataFrame({
    'y1': [0.53817, 0.52694, 0.5091, 0.48718, 0.49782, 0.4894],
    'y2': [0.30481, 0.2891, 0.26873, 0.21775, 0.20861, 0.20104],
    'y3': [0.1316, 0.13113, 0.10341, 0.08943, 0.09088, 0.07257],
    'y4': [0.13325, 0.10818, 0.10323, 0.10315, 0.0727, 0.06798],
    'y5': [1.10783, 1.05535, 0.98447, 0.89751, 0.87001, 0.83099],
    'y6': [0.07618, 0.07217, 0.06713, 0.05443, 0.05215, 0.05018],
    'y7': [0.28819, 0.27148, 0.18885, 0.15995, 0.1547, 0.08773]
}, index=samples)

# Run the pipeline
summary = fit_all_reselect(Y, X)

# Build a summary DataFrame
rows = []
for resp, info in summary.items():
    params = info['final_params']
    rows.append({
        'Response': resp,
        'Initial Model': info['initial_model_type'],
        'Initial R²': round(info['initial_r2'], 3),
        'Final Model': info['final_model_type'],
        'Final R²': round(info['final_r2'], 3),
        'Intercept': round(params.get('Intercept', np.nan), 3),
        'x1': round(params.get('x1', np.nan), 3),
        'x2': round(params.get('x2', np.nan), 3),
    })

df_summary = pd.DataFrame(rows)

# Display to the user
import ace_tools as tools; tools.display_dataframe_to_user("Transform-Reselection Fit Summary", df_summary)
