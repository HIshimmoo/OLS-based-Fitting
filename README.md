# LOO-CV Regression Script

This README explains how to use and understand the `loo_cv_regression.py` script, which performs leave-one-out cross-validated (LOO-CV) model selection over multiple variable subsets and transformations.

---

## 1. Introduction

* **Goal:** Automatically select the best combination of predictors (`x1, x2, …`) and data transforms (linear, log, exp, power, reciprocal) that maximizes out-of-sample predictive performance on one or more responses (`y1, y2, …`).
* **Core metric:** **LOO‑Q²**, the cross‑validated coefficient of determination:

  
    $$Q^2 = 1 - \frac{\sum_i (y_i - \hat y_{-i})^2}{\sum_i (y_i - \bar y)^2}$$  


  where $\hat y_{-i}$ is the prediction for $y_i$ from a model trained without the $i^{th}$ point.

---

## 2. Prerequisites

* **Python 3.7+**
* **Packages:**

  * `pandas`
  * `numpy`
  * `statsmodels`
  * `scikit-learn` (for `LeaveOneOut`)
  * `openpyxl` (Excel output)

Install via:

```bash
pip install pandas numpy statsmodels scikit-learn openpyxl
```

---

## 3. Usage

```bash
python loo_cv_regression.py --input input_data.xlsx --output results.xlsx [--transforms linear,log,...]
```

* `--input`: Excel file containing columns named `x1, x2, ...` and `y1, y2, ...`. If omitted, the script auto-detects the first `.xls/.xlsx` in the working directory.
* `--output`: Path for the generated Excel report (default: `regression_results_loo.xlsx`).
* `--transforms`: Comma-separated list of transforms to test (default: `linear,log,exp,power,reciprocal`).

---

## 4. Methodology

### 4.1. Data Transformations


|                         **Transform**                        | **What’s fitted**      | **Equivalent model in $(x,y)$**                             |
| :----------------------------------------------------------: | :--------------------- | :---------------------------------------------------------- |
|                          **linear**                          | fit $y$ vs $x$         | $\displaystyle y = \beta_0 + \beta_1\,x$                    |
|                            **log**                           | fit $\ln y$ vs $\ln x$ | $\displaystyle \ln y = \beta_0 + \beta_1\ln x$\\            |
|  ⇒ $\displaystyle y = e^{\beta_0}\,x^{\beta_1}$ (power‐law)  |                        |                                                             |
|                            **exp**                           | fit $\ln y$ vs $x$     | $\displaystyle \ln y = \beta_0 + \beta_1\,x$\\              |
| ⇒ $\displaystyle y = e^{\beta_0+\beta_1 x}=A\,e^{\beta_1 x}$ |                        |                                                             |
|                           **power**                          | fit $y$ vs $\ln x$     | $\displaystyle y = \beta_0 + \beta_1\ln x$ (semilog–$x$)    |
|                        **reciprocal**                        | fit $y$ vs $1/x$       | $\displaystyle y = \beta_0 + \beta_1\,\frac1x$ (hyperbolic) |

---

### Visually

* **Linear**: straight line in $(x,y)$.
* **Power‐law (log–log)**: straight line in $(\ln x,\ln y)$; a curve $y\propto x^b$ in $(x,y)$.
* **Exponential**: straight line in $(x,\ln y)$; an exponential curve in $(x,y)$.
* **Semilog-$x$**: straight line in $(\ln x,y)$; a logarithmic rise in $(x,y)$.
* **Reciprocal**: straight line in $(1/x,y)$; a $1/x$‐type decay in $(x,y)$.


### 4.2. Single-Variable Q² Heatmaps

For diagnostic purposes, the script computes **LOO‑Q²** for each pair `(y_j, x_i)` under each transform:

1. Hold out one sample, fit OLS on the remaining 5, predict the held-out.
2. Sum squared errors across all 6 holds, compute Q².
3. Repeat for every `x_i`, every `y_j`, every transform.

These results are written to Excel sheets named `Heatmap_Q2_<transform>` (rows = responses, columns = predictors), allowing you to spot which variables ever beat the naïve mean.

### 4.3. Exhaustive Subset Selection by Q²

In every case the code simply does an OLS regression of

$$
\underbrace{T_y(y)}_{\text{possibly }\ln y\text{ or }y}
\;=\;\beta_0 \;+\;\sum_i\beta_i\,T_{x}(x_i)
$$

with the five choices of $T_x,T_y$ tranformation

---

## 5. Output Summary (`Best_Model_Q2`)

The main summary sheet reports, for each response:

| Column                | Description                                                                        |
| --------------------- | ---------------------------------------------------------------------------------- |
| **Q² screening**      |  Q² for each transformation                                                        |
| **Best Transform**    | Heat map of optimal single-Variable Q²                                             |
| **Transform Details** | Human‑readable mapping (e.g. `X log(x); Y raw`)                                    |
| **Best Subset**       | Comma‑separated predictors chosen (e.g. `x2,x3`)                                   |
| **LOO‑Q2**            | Out‑of‑sample R² for that model                                                    |
| **R²**                | In‑sample fit R² (for reference)                                                   |
| **Formula**           | Linear equation in transformed inputs (e.g. `y = 0.49 +0.23*log(x2) -0.05*(1/x3)`) |

---

## 6. Interpretation and Customization

* **Q² cutoff:**  By default, no pre‑screening is enforced. To drop predictors that never beat the mean alone, filter on single‑x Q² (e.g. require `>0`) before subset search.
* **Field thresholds:**

  * Q² > 0.7: Excellent predictive power
  * Q² 0.5–0.7: Strong
  * Q² 0.2–0.5: Moderate
  * Q² ≤ 0.2: Low
* **Model simplicity:** If two models have nearly equal Q², prefer the one with fewer predictors for robustness.
* **Further extensions:** Incorporate LASSO/elastic‑net inside each LOO fold for automated shrinkage, or bootstrap‑based stability selection.
