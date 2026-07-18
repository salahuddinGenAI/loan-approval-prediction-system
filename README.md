# 🏦 Loan Approval Prediction System

> An end-to-end Machine Learning Classification project that predicts whether a loan application will be **Approved** or **Rejected** using applicant financial information and credit history.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blueviolet)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

# 📖 Project Overview

Loan approval is one of the most important decision-making processes in the banking and financial industry. Traditional manual loan evaluation can be time-consuming, inconsistent, and prone to human bias.

This project demonstrates an end-to-end Machine Learning workflow that predicts whether a loan application should be **Approved** or **Rejected** based on an applicant's financial profile and creditworthiness.

The project includes:

- Data Understanding
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Comparison
- Model Evaluation
- Best Model Selection
- Deployment Ready Model

---

# 🎯 Problem Statement

Financial institutions receive thousands of loan applications every day. Evaluating each application manually is expensive and inefficient.

The objective of this project is to build a Machine Learning classification model that helps banks identify eligible applicants by predicting loan approval status using historical applicant data.

---

# 💼 Business Impact

A successful prediction model can help financial institutions:

- Reduce loan processing time
- Improve decision consistency
- Minimize financial risk
- Identify high-risk applicants
- Support data-driven lending decisions
- Reduce manual workload for loan officers

---

# 📊 Dataset Information

**Source**

Kaggle Loan Approval Dataset

**Total Records**

- **4,269 Loan Applications**

**Total Features**

- **13 Columns**

**Target Variable**

```text
loan_status
```

- Approved
- Rejected

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| loan_id | Unique Loan ID |
| no_of_dependents | Number of Dependents |
| education | Education Level |
| self_employed | Employment Status |
| income_annum | Annual Income |
| loan_amount | Requested Loan Amount |
| loan_term | Loan Repayment Duration |
| cibil_score | Applicant Credit Score |
| residential_assets_value | Residential Asset Value |
| commercial_assets_value | Commercial Asset Value |
| luxury_assets_value | Luxury Asset Value |
| bank_asset_value | Bank Asset Value |
| loan_status | Loan Approval Status (Target) |

---

# ⚙️ Machine Learning Workflow

```
Problem Understanding
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
        ↓
Encoding
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Model Saving
        ↓
Deployment
```

---

# 🤖 Machine Learning Algorithms

Three supervised classification models were trained and evaluated.

## 1️⃣ Logistic Regression

- Baseline classification model
- Fast and interpretable
- Suitable for binary classification

---

## 2️⃣ Decision Tree Classifier

- Captures non-linear relationships
- Easy to understand and visualize

---

## 3️⃣ Random Forest Classifier

- Ensemble learning algorithm
- Reduces overfitting
- Provides higher predictive performance
- Used for final model selection (if it achieved the best results)

---

# 📈 Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|---------:|----------:|--------:|----------:|
| Logistic Regression | 90.82% | 91% | 91% | 91% |
| Random Forest | 95.92% | 96% | 96% | 96% |
| Decision Tree | 96.94% | 97% | 96% | 97% |

 
---

# 📷 Project Screenshots

## Dataset Preview

```
Images/dataset_preview.png
```

---

## Exploratory Data Analysis

```
images/eda.png
```

---

## Correlation Heatmap

```
images/heatmap.png
```

---

## Confusion Matrix

```
images/confusion_matrix.png
```

---

## Streamlit Application

```
images/streamlit_app.png
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Streamlit
- Git
- GitHub

---

# 📁 Project Structure

```
Loan-Approval-Prediction-System/

│

├── data/
│   └── loan_approval_dataset.csv

├── notebook/
│   └── Loan_Approval_Prediction_System.ipynb

├── models/
│   └── best_model.pkl

├── app/
│   └── streamlit_app.py

├── images/

│   ├── dataset_preview.png
│   ├── eda.png
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   └── streamlit_app.png

├── requirements.txt

├── README.md

└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/loan-approval-prediction-system.git
```

Navigate to the project directory

```bash
cd loan-approval-prediction-system
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the notebook

```bash
jupyter notebook
```

or launch the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

---

# 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Cross-validation for more reliable model evaluation
- Explainable AI using SHAP or LIME
- Interactive Streamlit dashboard
- REST API deployment using FastAPI
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Cloud deployment (Streamlit Community Cloud or Render)

---

# 👨‍💻 Author

**Mohammad Salahuddin Ayubi**

Aspiring Machine Learning Engineer passionate about building AI-powered solutions for real-world business problems.

- 💼 LinkedIn: *(Add your profile link)*
- 💻 GitHub: *(Add your GitHub profile)*

---

# ⭐ Support

If you found this project helpful, consider giving this repository a **⭐ Star** on GitHub.

It motivates me to continue building and sharing practical Machine Learning projects.
