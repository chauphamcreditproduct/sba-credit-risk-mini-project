# Predicting Five-Year Charge-Off Risk in SBA 7(a) Loans

## DATA 400 Individual Mini-Project

This project studies whether information available around loan approval can be used to predict whether an SBA 7(a) small-business loan will be charged off within five years.

### Research Question

**Using information available around loan approval, how well can we predict whether an SBA 7(a) loan will be charged off within five years?**

The project is a binary classification problem:

- `1` = documented charge-off within five years after approval
- `0` = no documented charge-off within the first five years

Loans without a complete five-year observation window are excluded from the modeling sample.

---

## Why This Project Matters

Credit-risk analysis is an important part of commercial banking and lending. Lenders need to identify loans that may have a higher chance of future loss while also avoiding unnecessary review of loans that are likely to perform well.

This project treats the model as a **risk-screening tool**, not as an automatic approval or rejection rule. The goal is to test how much predictive information is available in public SBA loan-level data.

Potential stakeholders include:

- commercial lenders
- credit analysts
- portfolio-risk teams
- SBA program stakeholders
- researchers studying small-business credit risk

---

## Data Source

The data come from the **U.S. Small Business Administration 7(a) & 504 FOIA dataset**:

https://data.sba.gov/dataset/7a-504-foia

The analysis uses the SBA 7(a) loan files:

- FY2010–FY2019
- FY2020–Present
- official SBA data dictionary

The source snapshot used in this project is **June 30, 2026**.

Because the target requires a complete five-year performance window, the latest eligible approval date is **June 30, 2021**.

One row represents one SBA 7(a) loan.

### Main Predictors

Examples of variables used in the model include:

- gross approval amount
- SBA guarantee percentage
- loan term
- initial interest rate
- jobs supported
- approval year
- project state
- business type
- business age
- processing method
- fixed or variable interest-rate indicator
- revolving status
- collateral indicator
- broad NAICS industry sector
- franchise indicator

Outcome-related fields such as loan status, charge-off date, charge-off amount, and paid-in-full date are used only to construct or verify the target. They are not used as predictors.

---

## Modeling Sample

After duplicate removal and target construction:

| Sample | Loans | Approval Period | 5-Year Charge-Off Rate |
|---|---:|---|---:|
| Training | 393,008 | Oct. 1, 2009 – Dec. 31, 2017 | 3.86% |
| Validation | 95,853 | Jan. 1, 2018 – Dec. 31, 2019 | 4.18% |
| Test | 54,249 | Jan. 1, 2020 – June 30, 2021 | 2.36% |

The first training observations appear in October 2009 because they are included in SBA Fiscal Year 2010.

The split is chronological rather than random so that the project more closely reflects a real credit-risk setting: learning from older loans and evaluating on later loan cohorts.

---

## Methods

The workflow includes:

1. data download and column standardization
2. duplicate removal and data-quality checks
3. construction of the five-year charge-off target
4. chronological train/validation/test split
5. exploratory data analysis
6. feature engineering
7. preprocessing with Scikit-learn pipelines
8. comparison of multiple classification models
9. validation-based threshold selection
10. learning-curve and generalization analysis
11. validation error analysis
12. one-time final test evaluation
13. feature-importance analysis

### Models Compared

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest

Because charge-offs are relatively rare, **PR-AUC / Average Precision** is used as the main model-comparison metric. ROC-AUC, precision, recall, F1 score, and confusion matrices are also reported.

---

## Main Results

### Validation Model Comparison

| Model | Validation PR-AUC |
|---|---:|
| Dummy Classifier | 0.042 |
| Logistic Regression | 0.159 |
| Decision Tree | 0.252 |
| Random Forest | **0.337** |

The **Random Forest** was selected because it had the highest validation PR-AUC.

The classification threshold was selected on the validation set by maximizing F1.

**Selected threshold: 0.690**

Validation performance at this threshold:

- Precision: **0.332**
- Recall: **0.557**
- F1: **0.416**
- False positives: **4,496**
- False negatives: **1,776**

### Final Test Results

The final model was refit using the combined training and validation data and then evaluated once on the locked 2020–June 2021 test set.

| Metric | Test Result |
|---|---:|
| PR-AUC | **0.352** |
| ROC-AUC | **0.931** |
| Precision | **0.426** |
| Recall | **0.314** |
| F1 | **0.362** |

Final test confusion matrix:

- True negatives: **52,427**
- False positives: **542**
- False negatives: **878**
- True positives: **402**

The model identified about **31.4% of actual five-year charge-offs**, while about **42.6% of loans flagged as high risk actually charged off within five years**.

---

## Exploratory Findings

Several descriptive patterns appeared in the training data:

- Loans below $50,000 had a five-year charge-off rate of about **5.59%**, compared with about **0.85%** for loans above $1 million.
- Loans with terms of 84 months or less had a charge-off rate of about **5.11%**, compared with about **0.48%** for terms above 180 months.
- Loans with initial interest rates between 8% and 11% had a charge-off rate of about **7.30%**, compared with about **1.35%** for rates below 5%.
- Charge-off rates also varied across broad NAICS sectors and SBA guarantee bands.

These are descriptive and predictive relationships, not causal effects.

---

## Most Predictive Features

For the final Random Forest, the largest feature importances were approximately:

1. loan term — **0.594**
2. initial interest rate — **0.078**
3. log loan amount — **0.076**
4. SBA guarantee percentage — **0.034**

Feature importance shows which variables were useful to the fitted model. It does not show that those variables caused charge-off.

---

## Repository Contents

```text
Sba-credit-risk-mini-project/
│
├── README.md
├── requirements.txt
│
├── DATA400_SBA_7a_Credit_Risk_Final.ipynb
├── DATA400_SBA_7a_Credit_Risk_Final.py
├── DATA400_SBA_7a_Credit_Risk_Report.pdf
├── SBA_7a_Credit_Risk.pptx
│
├── data/
│   └── README.md
│
├── figures/
│   └── README.md
│
└── output/
    └── README.md
```

The notebook is the main reproducible analysis. The Python file contains the same workflow in script form.

---

## How to Reproduce the Analysis

### 1. Clone this repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd Sba-credit-risk-mini-project
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Obtain the SBA data

Follow the instructions in:

```text
data/README.md
```

The raw SBA files are not included in this repository because they are large and are available directly from the official SBA data source.

### 4. Run the notebook

Open:

```text
DATA400_SBA_7a_Credit_Risk_Final.ipynb
```

and run all cells from top to bottom.

When using Google Colab, the notebook can save the raw data, figures, and outputs to Google Drive.

---

## Software / Packages

The analysis uses Python and the following main libraries:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- requests
- tqdm
- joblib
- openpyxl

See `requirements.txt` for the package list.

---

## Limitations

This project has several important limitations.

First, the public SBA data do not contain many borrower-level financial variables normally used in commercial underwriting, such as cash flow, debt-service coverage, leverage, liquidity, detailed credit history, or management quality.

Second, loans approved after June 30, 2021 cannot be used because the June 30, 2026 data snapshot does not provide a complete five-year performance window for those loans.

Third, the final test period includes COVID-era lending conditions, which may differ from earlier periods.

Fourth, charge-off is an observable credit outcome but is not identical to every possible definition of borrower default or financial distress.

Finally, the sample contains approved SBA 7(a) loans only. The results should not be generalized automatically to rejected applicants, all commercial borrowers, or interpreted as causal relationships.

---

## Conclusion

The results show that information available around SBA 7(a) loan approval contains useful predictive signal for five-year charge-off risk.

Random Forest performed best on the validation data and achieved a final test PR-AUC of **0.352** and ROC-AUC of **0.931**. However, performance at the selected threshold remained imperfect, especially in recall. This suggests that public loan-level characteristics can support credit-risk screening but cannot replace full underwriting information.

The model is therefore best interpreted as a **risk-screening and portfolio-analysis tool**, rather than an automatic lending decision rule.

---

## References

- U.S. Small Business Administration. **7(a) & 504 FOIA Loan Data.**  
  https://data.sba.gov/dataset/7a-504-foia

- Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn and PyTorch*. O'Reilly Media.

- Scikit-learn documentation.  
  https://scikit-learn.org/stable/

---

## Author

DATA 400 Individual Mini-Project  
Fall 2026
