# 🏦 Loan Approval Prediction System

An AI-powered web app that predicts whether a loan application will be **approved or rejected**, based on the applicant's income, credit score, assets, and other details.

Built with **Python, scikit-learn, and Streamlit** — you fill in a form, and the app instantly tells you the predicted decision along with a confidence score.

---

## ✨ What This Project Does

Banks look at a lot of factors before approving a loan — income, credit history, existing assets, and more. This project uses a machine learning model trained on real loan application data to automate that decision-making process.

You can try it yourself:
1. Enter applicant details (dependents, education, employment)
2. Enter financial details (income, loan amount, loan term, CIBIL score)
3. Enter asset details (residential, commercial, luxury, bank assets)
4. Click **Predict** and get an instant result 🎯

---

## 🖥️ App Preview

The app has a clean banking-style dashboard with:
- 🎨 A dark, premium theme
- 📋 A simple form to enter applicant details
- ✅❌ Color-coded approval/rejection cards
- 📊 A confidence gauge and risk level indicator
- 📈 A sidebar with model info and dataset stats

---

## 🧠 How It Works

1. **Data Cleaning** – The raw dataset was cleaned (removed extra spaces, fixed invalid values).
2. **Feature Encoding** – Text fields like *Education* and *Self Employed* were converted into numbers so the model can understand them.
3. **Scaling** – All numeric values were scaled to a common range using `StandardScaler`, so no single feature (like income) unfairly dominates the prediction.
4. **Model Training** – Three models were trained and compared:
   - Logistic Regression
   - Decision Tree
   - Random Forest ✅ (best performer)
5. **Model Selection** – Random Forest gave the highest and most consistent accuracy, so it was saved and used for the app.
6. **Prediction** – When you submit the form, the app applies the same encoding and scaling steps, then feeds your data into the trained model to get a prediction.

---

## 📊 Dataset

- **Source file:** `loan_approval_dataset.csv`
- **Total records:** ~4,270 loan applications
- **After cleaning:** 4,241 records used for training

**Columns in the dataset:**

| Column | Description |
|---|---|
| `no_of_dependents` | Number of people financially dependent on the applicant |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Applicant's annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan repayment period (in years) |
| `cibil_score` | Applicant's credit score (300–900) |
| `residential_assets_value` | Value of residential property owned |
| `commercial_assets_value` | Value of commercial property owned |
| `luxury_assets_value` | Value of luxury assets owned |
| `bank_asset_value` | Value of bank assets/savings |
| `loan_status` | Target column — Approved / Rejected |

---

## 🤖 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | 92.58% |
| Decision Tree | 96.94% |
| **Random Forest (Final Model)** | **97.78%** |

The final model was validated using 5-fold cross-validation, giving a consistent accuracy of **97.86% (± 0.44%)** — confirming it performs well on unseen data and isn't overfitting.

---

## 📁 Project Structure

```
├── __Loan_Approval_Prediction_System.ipynb   # Jupyter notebook: data cleaning, EDA, model training
├── loan_approval_dataset.csv                 # Raw dataset used for training
├── loan_approval_prediction_pipeline.pkl     # Saved trained model pipeline
├── streamlit_app.py                          # Streamlit web app
├── requirements.txt                          # Python dependencies
└── README.md                                 # You are here 📍
```

---

## 🚀 How to Run This Project Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/loan-approval-prediction-system.git
   cd loan-approval-prediction-system
   ```

2. **Install the required libraries**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open the link shown in your terminal (usually `http://localhost:8501`) 🎉

---

## 🌐 Live Demo

🔗 [Try the app here](#) *(add your Streamlit Community Cloud link once deployed)*

---

## 🛠️ Tech Stack

- **Python** – Core programming language
- **pandas / numpy** – Data handling
- **scikit-learn** – Model building and evaluation
- **Streamlit** – Web app framework
- **Jupyter Notebook** – Data exploration and model development

---

## ⚠️ Disclaimer

This project is built for **learning and portfolio purposes only**. Predictions from this app should **not** be used for actual loan or lending decisions.

---

## 🙋‍♂️ About Me

Made with ❤️ by **Salahud-Din Ayubi**
- 🔗 GitHub: [your-username](https://github.com/your-username)
- 🔗 LinkedIn: [your-profile](https://linkedin.com/in/your-profile)

If you found this project useful, consider giving it a ⭐ on GitHub!
