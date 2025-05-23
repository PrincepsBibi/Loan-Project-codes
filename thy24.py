# prompt: Can you use prety much the same code but this time instead of using the file /CleanedLoanData2.csv use /CleanedLoanData3.csv. This new file has all the same columns as before except it has a LoanTerm column. I want you to make this another parameter used to calculate someone's probability of defaulting

# Load the data from the CSV file 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

try:
    df = pd.read_csv("/Users/benjaminmashoko/Documents/HighSchool Senior Year/Internship/datasets/CleanedLoanData3.csv")
except FileNotFoundError:
    print("Error: 'CleanedLoanData3.csv' not found. Please make sure the file exists at the specified path.")
    exit()

# Print available columns
print("Available columns in the dataset:")
print(df.columns.tolist())

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("\nAfter dropping 'Unnamed: 0', remaining columns:")
    print(df.columns.tolist())

# Select only the columns we want to use that exist in the dataset
selected_columns = ['Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'LoanTerm', 'Default']
df = df[selected_columns]

# Convert categorical columns to numerical
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Assuming 'Default' is the target variable and other columns are features
try:
    X = df.drop('Default', axis=1)
    print("\nFeatures used for prediction (X columns):")
    print(X.columns.tolist())
    y = df['Default']
except KeyError as e:
    print(f"Error: Column '{e}' not found in the dataset. Please check your column names.")
    exit()

# Handle potential missing values
X.fillna(0, inplace=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=20)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

def get_loan_applicant_details():
    print("\nPlease enter the following loan applicant details:")
    
    # Define all the features we want to collect
    features = {
        'Income': 'Enter annual income: ',
        'InterestRate': 'Enter interest rate (as a decimal, e.g., 0.05 for 5%): ',
        'LoanAmount': 'Enter loan amount: ',
        'CreditScore': 'Enter credit score (300-850): ',
        'MonthsEmployed': 'Enter number of months employed: ',
        'LoanTerm': 'Enter loan term in months: ',
        'DTIRatio': 'Enter debt-to-income ratio (as a decimal, e.g., 0.35 for 35%): ',
        'HasMortgage': 'Do you have a mortgage? (Yes/No): '
    }
    
    applicant_data = {}
    
    for feature, prompt in features.items():
        while True:
            try:
                value = input(prompt).strip()
                
                if feature == 'HasMortgage':
                    value = value.capitalize()
                    if value not in ['Yes', 'No']:
                        print("Please enter either 'Yes' or 'No'")
                        continue
                    # Convert Yes/No to 1/0
                    value = 1 if value == 'Yes' else 0
                else:
                    value = float(value)
                    
                    # Add some basic validation
                    if feature == 'CreditScore' and (value < 300 or value > 850):
                        print("Credit score must be between 300 and 850")
                        continue
                    if feature == 'DTIRatio' and (value < 0 or value > 1):
                        print("DTI ratio must be between 0 and 1")
                        continue
                    if feature == 'InterestRate' and (value < 0 or value > 1):
                        print("Interest rate must be between 0 and 1")
                        continue
                
                applicant_data[feature] = value
                break
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a valid number.")
    
    return applicant_data

def predict_default():
    try:
        # Get applicant details
        applicant_data = get_loan_applicant_details()
        
        # Create a DataFrame with only the features that exist in our training data
        prediction_data = {col: applicant_data[col] for col in X.columns}
        new_customer = pd.DataFrame([prediction_data])
        
        # Make prediction
        default_probability = clf.predict_proba(new_customer)[:, 1][0]
        
        # Print results
        print("\nLoan Application Analysis:")
        print(f"Probability of default: {default_probability:.2%}")
        
        if default_probability > 0.5:
            print("Warning: High risk of default!")
        else:
            print("Low risk of default")
            
        # Print additional risk factors
        print("\nRisk Factors:")
        if applicant_data['CreditScore'] < 600:
            print("- Low credit score")
        if applicant_data['DTIRatio'] > 0.43:
            print("- High debt-to-income ratio")
        if applicant_data['MonthsEmployed'] < 12:
            print("- Short employment history")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
predict_default()