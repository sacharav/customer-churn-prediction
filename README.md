# Customer Churn Prediction

Independent machine learning project predicting customer churn for a telecom
provider, using the public [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM sample dataset, 7,043 customer records via Kaggle).

## Goal

Churn is expensive — it's usually cheaper to retain an existing customer
than acquire a new one. This project builds and compares models to identify
customers at high risk of leaving, and turns the findings into concrete
retention recommendations.

## Approach

- **Cleaning:** converted `TotalCharges` to numeric, dropped ~11 rows with
  missing values, removed the non-predictive `customerID` column.
- **Feature engineering:** added `AvgMonthlySpend` (total charges / tenure)
  and bucketed `tenure` into groups (0–12, 13–24, 25–48, 49–60, 60+ months)
  to capture how churn risk changes over a customer's lifecycle.
- **Encoding:** one-hot encoded all categorical features; 80/20 stratified
  train/test split.
- **Models compared:**
  - Logistic Regression (baseline, most interpretable)
  - k-Nearest Neighbors (k tuned from 1–20; k=20 performed best)
  - Neural Network (MLPClassifier, hyperparameters tuned via GridSearchCV)

## Results

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | ~79% | ~0.83 |
| k-Nearest Neighbors (k=20) | ~78% | ~0.82 |
| Neural Network (MLP) | ~79% | ~0.83 |

Logistic regression gave the best balance of performance and
interpretability — a meaningful advantage when the goal is explaining churn
drivers to a non-technical business audience, not just predicting them.

## Key churn drivers

- **Tenure** — shorter-tenure customers churn at much higher rates.
- **Contract type** — month-to-month contracts are strongly linked to churn.
- **Monthly charges / average spend** — higher spending correlates with higher churn.
- **Payment method** — electronic-check payers churn more than other methods.
- **Fiber optic internet** — this segment shows elevated churn.

## Business recommendations

1. Incentivize longer contracts (1–2 years) with loyalty discounts.
2. Use the model to flag high-risk customers for proactive retention outreach.
3. Bundle services for high-spending customers to increase perceived value.
4. Prioritize onboarding and support quality for new and fiber-optic customers.

## Tech stack

Python · Pandas · Scikit-learn · Seaborn · Matplotlib

## Running it

```bash
pip install pandas scikit-learn seaborn matplotlib
jupyter notebook customer_churn_prediction.ipynb
```

Make sure `Telco_Customer_Churn.csv` is in the same directory as the notebook.
