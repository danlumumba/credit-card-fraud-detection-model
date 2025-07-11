#  Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Given the highly imbalanced nature of fraud data, the project applies data preprocessing, resampling (SMOTE), and classification models including **Gradient Boosting** and **Random Forest** to build a reliable fraud detection pipeline.

---

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- The dataset contains transactions made by European cardholders in September 2013.
- **Features**: 30 (anonymized features `V1` to `V28`, `Time`, `Amount`)
- **Target**: `Class` — 0 (Non-Fraud), 1 (Fraud)

---

##  Machine Learning Workflow

###  1. Data Preprocessing
- Handled missing values
- Scaled numerical features (`Amount`, `Time`) using `StandardScaler`

###  2. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set to balance the fraud class.

###  3. Model Training
One model was trained and evaluated:
- **Random Forest Classifier**

###  4. Evaluation Metrics
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC-AUC Score**
- Focused on **Recall** and **Precision** for fraud cases due to class imbalance.

---

##  Results

###  Random Forest Classifier Performance

| Metric     | Non-Fraud (0) | Fraud (1) |
|------------|----------------|-----------|
| Precision  | 1.00           | 0.92      |
| Recall     | 1.00           | 0.78      |
| F1-Score   | 1.00           | 0.84      |
| Accuracy   | **1.00** overall |

- **Confusion Matrix**:
```
[[56650 6]
[ 20 70]]
```

- High **precision** on fraud means fewer false fraud alerts.
- Good **recall** on fraud ensures most frauds are caught.

---

##  Future Enhancements

###  1. Hyperparameter Tuning
- Use `GridSearchCV` or `RandomizedSearchCV` to optimize:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `class_weight`
- Example:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='f1', n_jobs=-1)

