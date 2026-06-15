# Customer-purchase-prediction


A machine learning model that predicts whether an e-commerce 
user will make a purchase, based on their browsing behavior 
and demographic data.

## Overview

Built a complete end-to-end ML pipeline — from raw data 
exploration to final predictions — using Python and scikit-learn. 
The model analyzes behavioral patterns like pages viewed, 
time on site, and past purchases to classify users as 
likely to convert or not.

## Dataset

Anonymized e-commerce user data with the following features:

| Feature | Description |
|---------|-------------|
| Age | User age |
| Income | Estimated annual income |
| City_Tier | City classification (1/2/3) |
| Pages_Viewed | Number of pages viewed per session |
| Products_Viewed | Number of products browsed |
| Time_On_Site | Session duration (minutes) |
| Previous_Purchases | Historical purchase count |
| Discount_Seen | Whether user saw a discount (0/1) |
| Browser_Version | Browser version identifier |
| Campaign_Code | Marketing campaign identifier |

**Target Variable:** `Converted` — 1 (purchased) or 0 (did not purchase)

> Note: Raw dataset not included in this repository.

## ML Pipeline

### 1. Exploratory Data Analysis
- Checked data distributions and class balance
- Identified missing values (~15% in Age, ~10% in Income)
- Found target imbalance: ~69% non-converted, ~31% converted

### 2. Preprocessing
- **Missing values:** Median imputation (fit on train, 
  applied to test — no data leakage)
- **Categorical columns:** Dropped for baseline 
  (Device_Type, Traffic_Source)
- **Feature scaling:** StandardScaler (fit on train, 
  applied to test)

### 3. Model
- **Algorithm:** Logistic Regression
- **Why:** Simple, interpretable, strong baseline 
  for binary classification
- **Parameters:** max_iter=1000, random_state=42

### 4. Evaluation
- **Metric:** F1 Score (accounts for class imbalance)
- **Training Accuracy:** 70.97%

## Key Findings

Most important features (by coefficient magnitude):
Pages_Viewed        →  0.461  (most influential)

Products_Viewed     →  0.315

Discount_Seen       →  0.265

Previous_Purchases  →  0.237

Time_On_Site        →  0.189

Income              →  0.107

**Insight:** Behavioral features (what the user does on 
the site) matter far more than demographic features 
(age, income) for predicting purchase intent.

## Results

| Metric | Score |
|--------|-------|
| Training Accuracy | 70.97% |
| F1 Score (validation) | ~0.69 |

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/saitejo/customer-purchase-prediction
cd customer-purchase-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place these files in the project directory:
train.csv

public_test.csv

private_test.csv

### 4. Run the notebook
```bash
jupyter notebook notebook.ipynb
```

Runs top to bottom, generates `submission.csv` with predictions.

## Tech Stack

- **Python 3.14**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **scikit-learn** — preprocessing, modeling, evaluation
- **Jupyter Notebook** — development environment

## Future Improvements

- [ ] Encode categorical features (Device_Type, Traffic_Source)
- [ ] Feature engineering (Pages_Viewed / Time_On_Site ratio)
- [ ] Try ensemble models (Random Forest, XGBoost)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Handle class imbalance with SMOTE

## Project Structure
customer-purchase-prediction/

├── notebook.ipynb      # Complete ML pipeline

├── submission.csv      # Final predictions

├── report.pdf          # Methodology summary

├── requirements.txt    # Python dependencies

├── .gitignore          # Excludes data files

└── README.md           # This file
