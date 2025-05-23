import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def get_user_input():
    print("\nPlease enter the following details for the loan applicant:")
    age = float(input("Age: "))
    income = float(input("Annual Income: "))
    loan_amount = float(input("Loan Amount: "))
    credit_score = float(input("Credit Score: "))
    months_employed = float(input("Months Employed: "))
    
    # Create feature interactions
    debt_to_income = loan_amount / income
    loan_to_income_ratio = loan_amount / (income / 12)  # Monthly income
    credit_age_ratio = credit_score / age
    
    return np.array([[age, income, loan_amount, credit_score, months_employed, 
                     debt_to_income, loan_to_income_ratio, credit_age_ratio]])

def predict_default_probability(model, scaler):
    user_data = get_user_input()
    scaled_data = scaler.transform(user_data)
    probability = model.predict_proba(scaled_data)[0][1]
    return probability

try:
    # Load and prepare data
    df = pd.read_csv("/Users/benjaminmashoko/Documents/HighSchool Senior Year/Internship/datasets/CleanedLoanData.csv")
    df = df.drop(columns=["Unnamed: 0"])

    # Handle any missing values
    df = df.fillna(df.mean())

    # Create new features
    df['DebtToIncome'] = df['LoanAmount'] / df['Income']
    df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['Income'] / 12)
    df['CreditAgeRatio'] = df['CreditScore'] / df['Age']
    df['IncomePerMonth'] = df['Income'] / 12
    df['LoanToCreditRatio'] = df['LoanAmount'] / df['CreditScore']

    # Select features
    X = df[["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "DebtToIncome", "LoanToIncomeRatio", "CreditAgeRatio", 
            "IncomePerMonth", "LoanToCreditRatio"]]
    y = df["Default"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create pipeline with SMOTE and Decision Tree
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__class_weight': ['balanced', None]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    print("\nTraining model with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Get best parameters
    print("\nBest parameters found:")
    print(grid_search.best_params_)

    # Get best model
    best_model = grid_search.best_estimator_

    # Perform cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc')
    print("\nCross-validation AUC scores:", cv_scores)
    print("Average CV AUC score: {:.4f}".format(cv_scores.mean()))

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print evaluation results
    print("\nModel Evaluation Results:")
    print("------------------------")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nDetailed Metrics:")
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))
    print("ROC AUC Score: {:.4f}".format(roc_auc))

    # Interactive prediction loop
    while True:
        try:
            default_probability = predict_default_probability(best_model, best_model.named_steps['scaler'])
            print(f"\nProbability of default: {default_probability:.2%}")
            
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
            print(f"An error occurred during prediction: {str(e)}")

except FileNotFoundError:
    print("Error: Could not find the dataset file. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    