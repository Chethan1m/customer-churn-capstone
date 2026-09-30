# Customer Churn Prediction and Segmentation

Capstone project for the Data Science with Python internship (Week 6).
An end-to-end pipeline: data acquisition, cleaning, EDA, supervised and unsupervised modelling, evaluation and recommendations.

## Dataset
IBM Telco Customer Churn (7,043 customers, 21 columns), included as `telco.csv`.
Source: https://github.com/IBM/telco-customer-churn-on-icp4d

## Approach
- **Cleaning:** fixed `TotalCharges` blanks (11 new customers with tenure 0), dropped `customerID`, encoded target.
- **Feature engineering:** `NumAddOns`, `AvgMonthlySpend`.
- **Supervised:** Logistic Regression, Random Forest, Gradient Boosting (5-fold CV, grid search, threshold tuning).
- **Unsupervised:** K-Means (k = 4) with PCA visualisation.

## Key results
| Item | Result |
|---|---|
| Churn rate | 26.5% |
| Best model | Tuned Gradient Boosting, test ROC-AUC 0.845 |
| Recall at tuned threshold (0.322) | 73.8% |
| Segments | 4 clusters, churn from 4% to 43% |

## Project structure
```
pipeline.py          full analysis script
telco.csv            dataset
figs/                generated charts
report/Capstone_Report.docx   full written report
requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python pipeline.py
```
Charts are saved in `figs/` and metrics are printed to the console.

## Author
Minuku Chethan Sai
