# 💳 Credit Card Fraud Detection using Logistic Regression

This project focuses on detecting fraudulent credit card transactions using **Logistic Regression** and **SMOTE** to handle class imbalance. It is based on the publicly available dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Logistic Regression](https://img.shields.io/badge/Model-Logistic%20Regression-lightgrey)
![SMOTE](https://img.shields.io/badge/Technique-SMOTE-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Problem Statement

Credit card fraud is a growing concern in the financial sector. The dataset is highly imbalanced with only **0.17%** fraudulent transactions. This project aims to build a robust model to detect frauds while minimizing false positives and false negatives.

## ⚙️ Tools & Technologies Used

- Python 3.11
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- seaborn, matplotlib

## 📂 Project Structure

```
credit_card_fraud_detection/
├── credit_card.py       # Main Python script
├── README.md            # Project documentation
├── .gitignore           # Ignore CSV and system files
└── creditcard.csv       # Dataset (not uploaded to GitHub)
```

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/srinivasgithub123/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Add Dataset

- Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Place `creditcard.csv` inside the project folder

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### 4. Run the Script

```bash
python credit_card.py
```

## 📊 Results & Evaluation

### Confusion Matrix

```
               Predicted
              0      1
Actual  0   83410   1885
        1      18    130
```

### Classification Report (on test data)

- Accuracy: 98%
- Recall (fraud class): 88%
- Precision (fraud class): 6%
- F1-Score (fraud class): 12%

### Fraud Detected:
Out of 148 fraud cases in test data, the model detected **130 correctly**.

## 📸 Sample Output (Console)

```
Predicted Fraud Transactions:
             Time       V1       V2  ...  Amount  Actual_Class  Predicted_Class
200304   133383.0  -1.02    1.62     ...   1.00             0                1
43204     41413.0 -15.14    7.37     ... 106.55             1                1
...
Detected actual frauds: 130 out of 492 total frauds in test data
```

## ❗ Notes

- The dataset is large and imbalanced — that's why **SMOTE** was applied **only to the training set**.
- The `.gitignore` file ensures that `creditcard.csv` is not pushed to GitHub to avoid size and licensing issues.

## 📄 License

The dataset used is provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), and its usage is subject to their terms.

## 👨‍💻 Author

**Srinivas Kankala**  
[GitHub Profile](https://github.com/srinivasgithub123)
