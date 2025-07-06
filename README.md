Here’s a **detailed README** for your code, covering the workflow, logic, and how each step operates.

---

# README: OLS Fitting with Multi-Model Screening and Automated Reporting

## Overview

This program automates multivariate regression analysis with intelligent model selection and screening. It is designed for scientific data (e.g., materials or process analysis) where you may have several predictors (x1, x2, ...) and several responses (y1, y2, ...). The script provides a transparent, reproducible workflow for variable selection, model fitting, and statistical reporting.

---

## Workflow & Fitting Logic

### **1. Input Handling**

* The program automatically searches the working directory for the first Excel file (unless `--input` is specified).
* It expects an Excel file with column names like `x1`, `x2`, ..., `y1`, `y2`, ... (case-insensitive).
* All `xN` columns are treated as predictors; all `yN` columns as responses.

---

### **2. Model Selection and Transform Testing**

* For each response (y):

  * The program fits regression models using all specified transforms for x (by default: `linear`, `log`, `exp`, `power`, and optionally `reciprocal`).
  * For each transform, the data are appropriately transformed and OLS regression is performed.
  * The transform/model yielding the highest R² (coefficient of determination) for the full multivariate fit is selected as the **best global transform** for y.

---

### **3. Predictor Screening (Single-X Screening)**

* For each x predictor:

  * A single-variable regression (`y ~ x`) is performed **using the globally selected transform**.
  * The p-value for each x in this regression is checked.
  * **Screening threshold:** If the p-value for x is less than the specified alpha (default: 0.05), x is retained for the final multivariate model for this y. Otherwise, it is excluded.
* The result is a **set of eligible predictors** for each response y.

---

### **4. Final Model Fitting**

* For each response y, a multivariate OLS regression is performed using only the predictors that passed the screening step, and using the selected transform.
* If no predictors are eligible, an intercept-only model is fit.
* The regression output includes coefficients, p-values for each coefficient, final model formula, R², and number of predictors used.

---

### **5. Best R² Fit (“Full-X” Model)**

* As an additional reference, for each response y, the program fits a multivariate model using **all x predictors** (regardless of p-value), again using the transform that maximizes R².
* This gives the "best possible" R² fit for y, allowing you to compare the effect of variable screening versus using all predictors.

---

### **6. Output & Reporting**

* All results are saved to an Excel file (`regression_results.xlsx` by default) with the following sheets:

  * **Summary:** Integrates model formulas, coefficients, p-values, transforms used, R², and number of predictors for each y. Also includes the “best R² fit” formulas and R² for each response.
  * **Screening P:** Shows the p-value of each x for each y in the screening step.
  * **\[Optional] Screening\_P\_\[transform]:** If configured, separate sheets with p-values for every (y, x) under every model type.
* The summary sheet provides an at-a-glance view of all fitted models and their statistical strength.

---

## **Customizing the Workflow**

* **Add new transforms:**
  Extend the `apply_transform` function to support additional model types (e.g., sqrt, log1p, polynomial).
* **Change screening threshold:**
  Use `--alpha` to set the significance level for predictor screening.
* **Select which transforms to try:**
  Use `--transforms` with a comma-separated list (e.g., `linear,log,reciprocal`) to control which model types are included.

---

## **Fitting Logic Explained**

1. **Transform selection ensures you’re not just fitting a linear model, but you are comparing a family of reasonable models (log-log, semi-log, etc.) for each response and picking the best.**
2. **Predictor screening uses statistical significance (p-value) in single-x regressions to avoid including noise variables and minimize overfitting.**
3. **Final fitting uses only the “screened” predictors, maximizing interpretability and avoiding collinearity problems caused by unnecessary variables.**
4. **A reference “full-x” model is provided so you can see what the theoretical best fit would be, even if it uses weak/noisy variables.**
5. **All outputs are transparent, exportable, and reviewable—no black boxes.**


