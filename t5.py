import zipfile
import pandas as pd
import numpy as np

# Machine Learning Model Architecture
from sklearn.ensemble import RandomForestClassifier

# Data Preprocessing and Feature Engineering
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

# Model Validation and Hyperparameter Optimization
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

# Pipelining and Data Transformation
from sklearn.pipeline import Pipeline as pipeline
from sklearn.compose import ColumnTransformer

# Model Evaluation Metrics
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Handling Imbalanced Datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the loan default dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing loan data
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor) - Preprocessed and split data
    """
    # Load the data with proper handling of the CSV format
    data = pd.read_csv("/Users/benjaminmashoko/Documents/HighSchool Senior Year/Internship/datasets/CleanedLoanData.csv", 
                      index_col=0)  # Use the first column as index
    
    # Print column names to verify
    print("Columns in dataset:", data.columns.tolist())
    
    # Ensure 'Default' column exists
    if 'Default' not in data.columns:
        raise ValueError("'Default' column not found in the dataset")
    
    # Separate numerical and categorical features
    numerical_features = data.select_dtypes(include=['int', 'float']).columns
    numerical_features = numerical_features.drop('Default')  # Remove Default from numerical features
    categorical_features = data.select_dtypes(include=['object']).columns
    
    print("Numerical features:", numerical_features.tolist())
    print("Categorical features:", categorical_features.tolist())
    
    # Create preprocessing pipelines
    numerical_pipeline = pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Split data into features and target
    X = data.drop(columns=['Default'])
    y = data['Default']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, preprocessor, data

def handle_imbalanced_data(X_train, y_train):
    """
    Handle class imbalance using SMOTE and RandomUnderSampler.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled) - Balanced dataset
    """
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Apply RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier with optimal hyperparameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using various metrics.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': range(X_test.shape[1]),
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.show()

def get_user_input(data):
    """
    Get user input for loan prediction.
    
    Parameters:
    -----------
    data : DataFrame
        Original dataset to get feature information
        
    Returns:
    --------
    DataFrame
        User input as a DataFrame
    """
    print("\nPlease enter the following information for loan prediction:")
    
    # Get numerical features
    numerical_features = data.select_dtypes(include=['int', 'float']).columns
    numerical_features = numerical_features.drop('Default')
    
    user_data = {}
    
    for feature in numerical_features:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                user_data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Get categorical features
    categorical_features = data.select_dtypes(include=['object']).columns
    
    for feature in categorical_features:
        unique_values = data[feature].unique()
        print(f"\nPossible values for {feature}: {', '.join(unique_values)}")
        while True:
            value = input(f"Enter {feature}: ")
            if value in unique_values:
                user_data[feature] = value
                break
            else:
                print(f"Please enter one of the valid values: {', '.join(unique_values)}")
    
    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])
    return user_df

def predict_default_probability(model, preprocessor, user_input):
    """
    Predict the probability of default for user input.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    preprocessor : ColumnTransformer
        Fitted preprocessor
    user_input : DataFrame
        User input data
        
    Returns:
    --------
    float
        Probability of default
    """
    # Preprocess user input
    processed_input = preprocessor.transform(user_input)
    
    # Get probability of default
    default_probability = model.predict_proba(processed_input)[0, 1]
    
    return default_probability

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, data = load_and_preprocess_data("/Users/benjaminmashoko/Documents/Internship/datasets/CleanedLoanData.csv")
    
    # Handle imbalanced data
    print("Handling class imbalance...")
    X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train)
    
    # Train model
    print("Training model...")
    model = train_model(X_train_resampled, y_train_resampled)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Get user input and make prediction
    while True:
        print("\nWould you like to predict the probability of default for a new loan? (yes/no)")
        choice = input().lower()
        
        if choice != 'yes':
            break
            
        user_input = get_user_input(data)
        probability = predict_default_probability(model, preprocessor, user_input)
        
        print(f"\nProbability of default: {probability:.2%}")
        if probability > 0.5:
            print("Warning: High risk of default!")
        else:
            print("Low risk of default.")

if __name__ == "__main__":
    main()
