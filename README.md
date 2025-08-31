# Loan E-Signing Prediction System  

A Machine Learning project to predict whether a customer will **digitally sign (e-sign)** a loan application based on their financial history and demographic details.  

## 🚀 Project Overview  
Banks and fintech companies often want to identify customers who are more likely to digitally sign loan applications. This project applies **Exploratory Data Analysis (EDA)**, **data preprocessing pipelines**, and **predictive modeling** to solve this problem.  

**Key Highlights:**  
- Conducted **EDA** using Seaborn & Matplotlib.  
- Built **data preprocessing pipelines** (handling missing values, encoding categorical variables, scaling numerical features).  
- Trained models using **Scikit-learn**.  
- Achieved **85% accuracy** using **Logistic Regression**.  

---

## 🛠️ Tech Stack  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn  

---

## 📊 Dataset  
- Contains customer **financial history, demographics, and loan details**.  
- **Target variable:** Whether a customer e-signed (`1 = Yes, 0 = No`).  
- Dataset used was a **simulated dataset for learning purposes**.  

---

## 📂 Project Workflow  

1. **Exploratory Data Analysis (EDA):**  
   - Visualized distributions (income, age, loan amount).  
   - Correlation analysis using heatmaps.  
   - Checked class balance (e-signed vs not).  

2. **Data Preprocessing:**  
   - Missing value handling (imputation).  
   - Encoding categorical variables (One-Hot Encoding).  
   - Feature scaling (StandardScaler/MinMaxScaler).  
   - Train-Test split (80/20).  

3. **Modeling:**  
   - Tried Logistic Regression, Decision Trees, Random Forest.  
   - **Logistic Regression** gave the best trade-off between performance and interpretability.  

4. **Evaluation Metrics:**  
   - Accuracy: **85%**  
   - Also checked: Precision, Recall, F1-score, ROC-AUC.  

5. **Insights:**  
   - Higher income & credit score → higher likelihood of e-signing.  
   - Previous loan defaults → lower chance of e-signing.  
   - Younger employed customers → more likely to adopt digital signing.  

---

## 📈 Results  
- Final Model: **Logistic Regression**  
- Accuracy: **85%**  
- Provided interpretable coefficients to understand **feature impact**.  

---

## 🔮 Future Improvements  
- Implement advanced models (Random Forest, XGBoost).  
- Perform hyperparameter tuning (GridSearchCV).  
- Feature engineering & dimensionality reduction (PCA).  
- Deploy model via **Flask/FastAPI** for real-time predictions.  

---

## 📌 How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/loan-esigning-prediction.git
   cd loan-esigning-prediction

2.Install dependencies:pip install -r requirements.txt


3.Run Jupyter Notebook / Python script:
  jupyter notebook Loan_Esigning_Prediction.ipynb





📚 Requirements
Python 3.7+
pandas
numpy
seaborn
matplotlib
scikit-learn
Install all with:
pip install -r requirements.txt
📜 License
This project is for educational purposes only and uses simulated data.

---

👉 You can just rename your notebook to `Loan_Esigning_Prediction.ipynb` and create a `requirements.txt` with:  
pandas
numpy
seaborn
matplotlib
scikit-learn
