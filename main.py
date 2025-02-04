import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb 
from sklearn.metrics import classification_report, accuracy_score

# Load the credit card fraud detection dataset
dataset = pd.read_csv('creditcard.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(dataset.head())


# Check for missing values
print("\nMissing Values:")
print(dataset.isnull().sum())

# Check data types and basic info
print("\nDataset Info:")
print(dataset.info())

# Select 450 legitimate and 450 fraudulent transactions
legit_df = dataset[dataset['Class'] == 0].sample(n=450, random_state=42)
fraud_df = dataset[dataset['Class'] == 1].sample(n=450, random_state=42)

# Combine the two DataFrames
dataset_combined = pd.concat([legit_df, fraud_df], ignore_index=True)

# Define features (X) and target (y)
X = dataset_combined.drop(columns=['Class'])  # Features (exclude target column)
y = dataset_combined['Class']  # Target variable (fraud or legit)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(dataset_combined['Class'].value_counts())
# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Train Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions using Decision Tree
y_pred_tree = decision_tree_model.predict(X_test)

# Evaluate the Decision Tree model
print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_tree))


# Train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Make predictions using XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))
