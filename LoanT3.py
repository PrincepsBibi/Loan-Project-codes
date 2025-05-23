import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def get_user_input():
    print("\nPlease enter the following details for the loan applicant:")
    age = float(input("Age: "))
    income = float(input("Annual Income: "))
    loan_amount = float(input("Loan Amount: "))
    credit_score = float(input("Credit Score: "))
    months_employed = float(input("Months Employed: "))
    
    return np.array([[age, income, loan_amount, credit_score, months_employed]])

def predict_default_probability(model, scaler):
    user_data = get_user_input()
    scaled_data = scaler.transform(user_data)
    probability = model.predict_proba(scaled_data)[0][1]
    return probability

df = pd.read_csv("/Users/benjaminmashoko/Documents/HighSchool Senior Year/Internship/datasets/CleanedLoanData3.csv")

df = df.drop(columns=["Unnamed: 0"])

X = df[["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed"]]
y = df["Default"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression with balanced class weight
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Print evaluation results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Interactive prediction loop
while True:
    try:
        default_probability = predict_default_probability(model, scaler)
        print(f"\nProbability of default: {default_probability:.2%}")
        
        # Calculate and display model accuracy
        model_accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy on test data: {model_accuracy:.2%}")
        
        if default_probability > 0.5:
            print("Warning: High risk of default!")
        else:
            print("Low risk of default")
            
        another_prediction = input("\nWould you like to make another prediction? (yes/no): ").lower()
        if another_prediction != 'yes':
            break
            
    except ValueError:
        print("Please enter valid numerical values for all fields.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")