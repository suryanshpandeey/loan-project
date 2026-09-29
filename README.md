# 🏦 Loan Approval Prediction

A machine learning classification project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial, demographic, and asset-related information.

The project compares **Logistic Regression** and **Random Forest**, explores feature selection and manual feature engineering, and evaluates the effect of **PCA-based dimensionality reduction**.

---

## 📌 Project Overview

Loan approval decisions depend on several factors such as credit score, income, loan amount, loan term, and the applicant's assets.

The objective of this project is to build a classification model that can predict loan approval status while systematically comparing different modeling approaches and feature representations.

### Key objectives

* Perform exploratory data analysis (EDA)
* Clean and preprocess the dataset
* Encode categorical variables
* Split the data into training, validation, and testing sets
* Build Logistic Regression and Random Forest models
* Tune classification thresholds for Logistic Regression
* Perform Random Forest hyperparameter tuning
* Select features using correlation with the target
* Perform manual feature engineering
* Evaluate PCA as a dimensionality-reduction technique
* Train the final model using training + validation data
* Evaluate the final model once on an untouched test set

---

## 📊 Dataset

The project uses the `loan_approval_dataset.csv` dataset.

### Dataset size

* **4,269 loan applications**
* **12 columns initially**
* **11 predictive features after removing `loan_id`**
* **No missing values**
* **No duplicate rows**

The target variable is:

```text
loan_status
```

with two classes:

* `Approved`
* `Rejected`

The dataset contains approximately:

* **62.22% Approved**
* **37.78% Rejected**

---

## 🧾 Features

| Feature                    | Description                            |
| -------------------------- | -------------------------------------- |
| `no_of_dependents`         | Number of dependents                   |
| `education`                | Applicant's education status           |
| `self_employed`            | Whether the applicant is self-employed |
| `income_annum`             | Annual income                          |
| `loan_amount`              | Requested loan amount                  |
| `loan_term`                | Loan term                              |
| `cibil_score`              | Applicant's CIBIL/credit score         |
| `residential_assets_value` | Value of residential assets            |
| `commercial_assets_value`  | Value of commercial assets             |
| `luxury_assets_value`      | Value of luxury assets                 |
| `bank_asset_value`         | Value of bank assets                   |
| `loan_status`              | Target variable                        |

The `loan_id` column is removed because it is only an identifier and is not intended to provide predictive information.

---

## 🔍 Exploratory Data Analysis

The project performs EDA before model training, including:

* Dataset shape and data types
* Descriptive statistics
* Duplicate detection
* Missing-value analysis
* Target-class distribution
* Numerical feature distributions
* Categorical feature analysis
* Approval rates across categories
* Numerical features vs. loan status
* Correlation analysis

One notable finding is that the target is moderately imbalanced, with approved applications representing about 62% of the dataset.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

1. Removing unnecessary whitespace from column names
2. Cleaning categorical values
3. Removing `loan_id`
4. Encoding categorical variables
5. Separating features and target
6. Scaling features where required
7. Performing a stratified train/validation/test split

### Dataset split

The data is divided into:

```text
70% Training
15% Validation
15% Testing
```

The split is stratified and uses:

```python
random_state=42
```

The validation set is used for model and hyperparameter selection, while the test set remains untouched until final evaluation.

---

# 🤖 Models

## 1. Logistic Regression

Logistic Regression is used as the linear baseline model.

Instead of relying only on the default probability threshold of `0.5`, thresholds from:

```text
0.1 → 0.9
```

are evaluated using the validation set.

The best baseline Logistic Regression threshold was **0.4**, achieving approximately **93.75% validation accuracy**.

---

## 2. Random Forest

Random Forest is used as the main tree-based model.

The project evaluates different combinations of:

* `n_estimators`
* `max_depth`
* `min_samples_split`

The search is performed using the validation set rather than the final test set.

---

# 🎯 Feature Selection

Features are ranked according to their **absolute correlation with the target** using the training data.

The strongest correlation was observed for:

```text
cibil_score
```

with a correlation magnitude of approximately:

```text
0.764
```

Other features had substantially smaller correlations.

The project then evaluates models using the top:

```text
3 features
5 features
7 features
9 features
```

### Validation results

| Features | Logistic Regression | Random Forest |
| -------: | ------------------: | ------------: |
|    Top 3 |              93.44% |        95.31% |
|    Top 5 |              93.59% |        95.78% |
|    Top 7 |              93.59% |        96.56% |
|    Top 9 |              93.75% |        98.28% |

The results show that Random Forest benefits substantially from retaining more of the correlated features.

---

# 🛠️ Feature Engineering

The project introduces two manually engineered features.

### 1. Total Asset Value

The four individual asset columns are combined:

```python
total_asset_value =
    residential_assets_value
    + commercial_assets_value
    + luxury_assets_value
    + bank_asset_value
```

### 2. Loan-to-Income Ratio

A new financial ratio is created:

```python
loan_to_income = loan_amount / income_annum
```

The original asset, loan, and income columns are then removed from this engineered representation.

This creates a more compact representation of the applicant's overall financial position and loan burden.

---

# 📉 PCA

Principal Component Analysis (PCA) is also investigated as a dimensionality-reduction technique.

The project evaluates:

```text
3 components
5 components
7 components
```

for both Logistic Regression and Random Forest.

Different Random Forest hyperparameters and Logistic Regression probability thresholds are evaluated on the validation set.

---

# 📈 Model Comparison

The major validation results are:

| Approach                  | Logistic Regression | Random Forest |
| ------------------------- | ------------------: | ------------: |
| Baseline — all features   |              93.75% |        98.59% |
| Top 9 correlated features |              93.75% |        98.28% |
| Engineered features       |              87.66% |    **99.84%** |
| PCA — 7 components        |              94.53% |        95.78% |

These results show that the manually engineered representation produced a particularly strong validation result for Random Forest, while PCA did not improve the tree-based model.

---

# 🏆 Final Model

The final model is a **Random Forest classifier trained using the engineered features**.

### Configuration

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
```

After selecting the final configuration, the model is retrained using:

```text
Training + Validation data
```

The final evaluation is then performed on the previously untouched test set.

### Final Test Accuracy

**99.38%**

```text
Test Accuracy: 0.9937597503900156
```

This corresponds to approximately **99.4% test accuracy**.

---

# 💡 Key Findings

### 1. CIBIL score is highly informative

`cibil_score` has the strongest correlation with the loan approval target among the features examined.

### 2. Random Forest performs strongly

Random Forest consistently achieves higher validation accuracy than Logistic Regression across the feature-selection experiments.

### 3. Feature engineering matters

Combining the four asset variables into `total_asset_value` and creating `loan_to_income` substantially improved the Random Forest validation performance.

### 4. PCA did not improve the final model

The PCA experiments produced lower validation accuracy for Random Forest than the engineered-feature approach.

### 5. Test data was kept separate

The final Random Forest was trained on the combined training and validation sets and evaluated only once on the untouched test set.

---

# 🗂️ Project Structure

```text
loan-approval-prediction/
│
├── loan_approval_project_presentable.ipynb
├── loan_approval_dataset.csv
└── README.md
```

---

# 🧰 Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical computation
* **Matplotlib** — visualization
* **Seaborn** — statistical visualization
* **Scikit-learn** — preprocessing and machine learning

### Machine Learning techniques

* Logistic Regression
* Random Forest
* Correlation-based Feature Selection
* Manual Feature Engineering
* Principal Component Analysis (PCA)
* Probability Threshold Tuning
* Hyperparameter Search
* Stratified Train/Validation/Test Split

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
loan_approval_project_presentable.ipynb
```

Make sure `loan_approval_dataset.csv` is placed in the appropriate dataset directory or update the dataset path in the notebook.

---

