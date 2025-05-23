import zipfile
# Module to handle extraction and management of ZIP compressed files.

import pandas as pd
# Primary library for structured data handling using DataFrames.

import numpy as np
# Essential package for numerical computing and array operations.

# =============================================================================
# MACHINE LEARNING MODEL ARCHITECTURE
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
# Ensemble learning method: Builds multiple decision trees and merges their results for better accuracy and stability.

# =============================================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
# - StandardScaler: Normalize features by removing the mean and scaling to unit variance.
# - OneHotEncoder: Encode categorical features as a one-hot numeric array.
# - PolynomialFeatures: Generate new polynomial and interaction features to capture non-linear relationships.

# =============================================================================
# MODEL VALIDATION AND HYPERPARAMETER OPTIMIZATION
# =============================================================================

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
# - train_test_split: Partition data into training and testing sets.
# - cross_val_score: Evaluate model performance via cross-validation.
# - GridSearchCV: Exhaustive search over specified parameter values for an estimator.

# =============================================================================
# PIPELINING AND DATA TRANSFORMATION
# =============================================================================

from sklearn.pipeline import Pipeline as pipeline
# Pipeline: Assemble several steps (preprocessing + model) into one sequential object.

from sklearn.compose import ColumnTransformer
# Apply different preprocessing pipelines to different feature columns.

# =============================================================================
# MODEL EVALUATION METRICS
# =============================================================================

from sklearn.metrics import accuracy_score, make_scorer
# - accuracy_score: Calculates the ratio of correctly predicted observations to total observations.
# - make_scorer: Create a custom scoring function for model evaluation or hyperparameter tuning.

# =============================================================================
# HANDLING IMBALANCED DATASETS
# =============================================================================

from imblearn.over_sampling import RandomOverSampler, SMOTE
# - RandomOverSampler: Duplicate random records from the minority class to balance the dataset.
# - SMOTE (Synthetic Minority Over-sampling Technique): Create synthetic examples of the minority class.

from imblearn.under_sampling import RandomUnderSampler
# Randomly remove samples from the majority class to balance the class distribution.

# Define the file path for the input dataset.
file_location = 'Loan_default.csv'  # Updated to use local file path

# Load the CSV data directly into a pandas DataFrame.
Data = pd.read_csv(file_location)

# Preview the first few rows to confirm successful loading.
print("First few rows of the dataset:")
print(Data.head())

# Drop the 'LoanID' column as it is an identifier and not useful for prediction.
Data = Data.drop(columns='LoanID')

# Verify that the column has been dropped by checking the DataFrame shape.
print("Updated Data Shape:", Data.shape)

# Select numerical columns (int and float types) for further analysis and model training.
num = Data.select_dtypes(include=['int', 'float'])

# Display the selected numerical features to verify.
print("Selected Numerical Features:", num.columns.tolist())

def ColumnTrans(cat):
    """
    Function to convert categorical variables into numerical variables
    by mapping unique values to integer indices.

    Parameters:
    ----------
    cat : DataFrame
        A pandas DataFrame containing categorical columns to be transformed.

    Returns:
    -------
    cat : DataFrame
        The original DataFrame with categorical columns transformed into numerical values.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    cat_transformed = cat.copy()
    
    # Iterate over each column in the DataFrame.
    for column in cat_transformed.columns:
        # Get unique values for the column
        unique_values = cat_transformed[column].unique()
        
        # Create a mapping of each unique value to a corresponding integer
        value_map = {value: index for index, value in enumerate(unique_values)}
        
        # Map the column's categorical values to their integer indices
        cat_transformed[column] = cat_transformed[column].map(value_map)
    
    return cat_transformed

# Select the categorical columns (object type) from the DataFrame.
cat = Data.select_dtypes(include='object')

# Apply the custom ColumnTrans function to transform categorical columns into numerical format.
cat_transformed = ColumnTrans(cat)

# Verify the transformation by displaying the first few rows of the transformed categorical data.
print("Transformed Categorical Columns (first few rows):")
print(cat_transformed.head())

# Concatenate the numerical and transformed categorical features along the columns (axis=1).
df = pd.concat([num, cat_transformed], axis=1)

# Verify the combined DataFrame by displaying the first few rows.
print("Combined DataFrame (first few rows):")
print(df.head())

# Separate features (X) from the target variable (y)
x1 = df.drop(columns='Default')  # Features: All columns except 'Default'
y1 = df['Default']  # Target: 'Default' column

# Verify the separation by displaying the shapes of the features and target.
print("Shape of Features (X):", x1.shape)
print("Shape of Target (y):", y1.shape)

# Initialize the resampling techniques
ros = RandomOverSampler()  # Random Over-Sampling to balance the dataset by increasing the minority class.
rus = RandomUnderSampler()  # Random Under-Sampling to balance the dataset by decreasing the majority class.
smote = SMOTE()  # SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples for the minority class.

# Verify that the resampling methods are correctly initialized
print("Resampling techniques initialized:")
print("RandomOverSampler:", ros)
print("RandomUnderSampler:", rus)
print("SMOTE:", smote)

# Apply Random Over-Sampling to increase the minority class
x2, y2 = ros.fit_resample(x1, y1)
print("Shape after RandomOverSampler (ROS):", x2.shape, y2.shape)

# Apply SMOTE to generate synthetic samples for the minority class
x3, y3 = smote.fit_resample(x2, y2)
print("Shape after SMOTE:", x3.shape, y3.shape)

# Apply Random Under-Sampling to decrease the majority class
x, y = rus.fit_resample(x3, y3)
print("Shape after RandomUnderSampler (RUS):", x.shape, y.shape)

# Final balanced dataset
print("Final Balanced Dataset Shape (X):", x.shape)
print("Final Balanced Dataset Shape (Y):", y.shape)

# Split the balanced dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets to verify the split
print("Shape of Training Features (X_train):", x_train.shape)
print("Shape of Testing Features (X_test):", x_test.shape)
print("Shape of Training Target (y_train):", y_train.shape)
print("Shape of Testing Target (y_test):", y_test.shape)

# Initialize the Random Forest Classifier with 2000 estimators (trees)
model = RandomForestClassifier(n_estimators=2000, random_state=42)

# Display the model's parameters to verify the configuration
print("Random Forest Classifier initialized with parameters:")
print(model)

# Train the model using the training data
model.fit(x_train, y_train)

# Display a message to indicate that training is complete
print("Model training complete with Random Forest Classifier.")

# Use the trained model to make predictions on the test data
prediction = model.predict(x_test)

# Display the shape of the predictions to verify
print("Shape of predictions:", prediction.shape)

# Since the data was imbalanced looking to see if the model predicts both values or needs more work.
(prediction == 0).sum()

(prediction == 1).sum()

# Import the evaluation metrics from sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)

# Display the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
