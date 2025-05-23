# prompt: Can you use prety much the same code but this time instead of using the file /CleanedLoanData2.csv use /CleanedLoanData3.csv. This new file has all the same columns as before except it has a LoanTerm column. I want you to make this another parameter used to calculate someone's probability of defaulting

# Load the data from the CSV file 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


try:
    df = pd.read_csv("/Users/benjaminmashoko/Documents/HighSchool Senior Year/Internship/datasets/CleanedLoanData3.csv")
except FileNotFoundError:
    print("Error: 'CleanedLoanData3.csv' not found. Please make sure the file exists at the specified path.")
    exit() # Exit the script if the file is not found


# Assuming 'Default' is the target variable and other columns are features
# Adjust column names if necessary based on your CSV file
try:
    X = df.drop('Default', axis=1)
    X = X.drop('Unnamed: 0', axis=1) # Drop unnamed column if present
    y = df['Default']
except KeyError as e:
    print(f"Error: Column '{e}' not found in the dataset. Please check your column names.")
    exit()

# Handle potential missing values (replace with more suitable imputation if needed)
X.fillna(0, inplace=True)  # Fill NaN values with 0


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=20)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Function to predict default probability for a new customer
def predict_default():
    try:
        # Get input from user for each feature
        # Replace with the actual feature names from your dataset.  
        feature_names = X.columns.tolist() # Get feature names from dataframe
        new_customer_data = {}
        for feature in feature_names:
          while True:
            try:
              value = float(input(f"Enter {feature}: ")) # Use float to handle potential decimals
              new_customer_data[feature] = value
              break
            except ValueError:
              print(f"Invalid input for {feature}. Please enter a number.")

        new_customer = pd.DataFrame([new_customer_data])  # Create dataframe for prediction

        # Ensure the columns in new_customer match the training data
        missing_cols = set(X.columns) - set(new_customer.columns)
        for c in missing_cols:
          new_customer[c] = 0 # Fill missing columns with 0

        # Ensure column order matches training data
        new_customer = new_customer[X.columns]


        default_probability = clf.predict_proba(new_customer)[:, 1][0]
        print(f"Probability of default: {default_probability}")
    except ValueError:
        print("Invalid input. Please enter numbers only.")
    except KeyError as e:
        print(f"Error: Feature '{e}' not found in dataset. Check your input and feature names.")


# Example usage
predict_default()
