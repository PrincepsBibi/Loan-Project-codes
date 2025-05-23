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

def load_and_preprocess_data(file_path="/Users/benjaminmashoko/Documents/Internship/datasets/Loan_default.csv"):
    """
    Load and preprocess the loan data with additional features
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Select features including additional relevant ones
    feature_columns = [
        'Age', 'Income', 'CreditScore', 'LoanAmount', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
    ]
    
    # Create dummy variables for categorical columns
    categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Combine numerical and encoded categorical features
    X = pd.concat([
        df_encoded[feature_columns],
        df_encoded[[col for col in df_encoded.columns if any(cat in col for cat in categorical_columns)]]
    ], axis=1)
    
    y = df['Default']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler, X.columns

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance
    """
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(model.coef_[0])
    })
    importance = importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance.head(15), x='Importance', y='Feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def train_model(X, y):
    """
    Train the model with class balancing and cross-validation
    """
    # Calculate class weights
    class_weights = dict(enumerate(class_weight.compute_class_weight(
        'balanced', classes=np.unique(y), y=y
    )))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train the model with class weights
    model = LogisticRegression(
        random_state=42,
        class_weight=class_weights,
        max_iter=1000,
        solver='lbfgs'
    )
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train the final model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, X_test, y_test

def predict_default(model, scaler, feature_names, **features):
    """
    Make a prediction for a new client
    """
    # Create a DataFrame with the input features
    input_df = pd.DataFrame([features])
    
    # Ensure all feature columns exist (fill missing with 0 for dummy variables)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]
    
    return prediction, probability

def main():
    try:
        # Load and preprocess data
        data_path = '/Users/benjaminmashoko/Documents/Internship/datasets/Loan_default.csv'
        X, y, scaler, feature_names = load_and_preprocess_data(data_path)
        
        # Train the model
        model, X_test, y_test = train_model(X, y)
        
        # Plot feature importance
        plot_feature_importance(model, feature_names)
        
        # Save the model and scaler
        joblib.dump(model, 'loan_default_model.joblib')
        joblib.dump(scaler, 'loan_default_scaler.joblib')
        joblib.dump(feature_names, 'feature_names.joblib')
        
        # Example prediction
        example_features = {
            'Age': 35,
            'Income': 50000,
            'CreditScore': 700,
            'LoanAmount': 10000,
            'MonthsEmployed': 24,
            'NumCreditLines': 3,
            'InterestRate': 5.5,
            'LoanTerm': 36,
            'DTIRatio': 0.28
        }
        
        prediction, probability = predict_default(model, scaler, feature_names, **example_features)
        
        print(f"\nPrediction for new client:")
        print(f"Will default: {'Yes' if prediction == 1 else 'No'}")
        print(f"Probability of default: {probability:.2%}")
        
    except FileNotFoundError:
        print("Error: Loan_default.csv file not found. Please ensure the data file exists at the specified path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 