from flask import Flask, render_template, request, make_response
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('creditcard.csv')
duplicates_count_before = dataset.duplicated().sum()

dataset = dataset.drop_duplicates()

# 450 legitimate and 450 fraudulent transactions
legit_df = dataset[dataset['Class'] == 0].sample(n=450, random_state=42)
fraud_df = dataset[dataset['Class'] == 1].sample(n=450, random_state=42)

# Combine the two DataFrames
dataset_combined = pd.concat([legit_df, fraud_df], ignore_index=True)

# Define features (X) and target (y)
X = dataset_combined.drop(columns=['Class'])
y = dataset_combined['Class']

# Apply Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Dictionary to store results
results = {}

def generate_plot():
    """Generate a plot and return it as a base64 encoded string."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_metrics(metrics):
    """Generate a bar plot of evaluation metrics and return it as a base64 encoded string."""
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'green', 'coral', 'orange'])
    plt.title('Model Evaluation Metrics', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.ylim(0, 1)  # Metrics are between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a buffer and return it as a base64 string
    return generate_plot()

def plot_confusion_matrix(y_true, y_pred):
    """Generate a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    plt.xticks([0, 1], ['Legitimate', 'Fraudulent'], fontsize=14)
    plt.yticks([0, 1], ['Legitimate', 'Fraudulent'], fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=16)
    return generate_plot()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    algorithm = request.form['algorithm']
    model = None

    if algorithm == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
        model_name = "Logistic Regression"
    elif algorithm == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
        model_name = "Decision Tree"
    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model_name = "XGBoost"

    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        selected_metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
            'Recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
            'F1-Score': classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
        }

        results[model_name] = {
            'accuracy': selected_metrics['Accuracy'],
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        selected_result = {
            'algorithm': model_name,
            'accuracy': selected_metrics['Accuracy'],
            'precision': selected_metrics['Precision'],
            'recall': selected_metrics['Recall'],
            'f1_score': selected_metrics['F1-Score'],
            'plot': plot_metrics(selected_metrics),  # Use the new plot_metrics function
            'confusion_matrix_plot': plot_confusion_matrix(y_test, y_pred),
        }

        return render_template('index.html', selected_result=selected_result, results=results)
    else:
        return "Invalid algorithm selected", 400

@app.route('/preprocessing')
def preprocessing():
    # Dataset Info
    dataset_info = {
        "entries": len(dataset),
        "columns": len(dataset.columns)
    }

    # Missing Values
    missing_values = dataset.isnull().sum().to_dict()
    missing_values_check = "No missing values found." if sum(missing_values.values()) == 0 else "Missing values found."

    # Duplicates
    duplicates_count_after = dataset.duplicated().sum()  # Duplicates after removal
    duplicates_check = f"{duplicates_count_before} duplicates found and removed." if duplicates_count_before > 0 else "No duplicates found."

    # Data After Removing Duplicates
    data_after_duplicates = {
        "entries": len(dataset),
        "columns": len(dataset.columns)
    }

    # Data Before Standard Scaler
    X_before_scaler = dataset_combined.drop(columns=['Class'])

    # Data After Standard Scaler
    # Convert X_scaled (NumPy array) to a Pandas DataFrame
    X_after_scaler_df = pd.DataFrame(X_scaled, columns=X.columns)
    # Convert the DataFrame to a list of dictionaries for Jinja2 rendering
    X_after_scaler = X_after_scaler_df.head(10).to_dict(orient='records')

    # Data Before SMOTE
    # Calculate class distribution before SMOTE
    class_dist_before_smote = {
        "legitimate": len(y[y == 0]),
        "fraudulent": len(y[y == 1])
    }

    # Data After SMOTE
    # Calculate class distribution after SMOTE
    class_dist_after_smote = {
        "legitimate": len(y_resampled[y_resampled == 0]),
        "fraudulent": len(y_resampled[y_resampled == 1])
    }

    # Data Before Train-Test Split
    data_before_split = {
        "total_samples": len(X_resampled),
        "legitimate": len(y_resampled[y_resampled == 0]),
        "fraudulent": len(y_resampled[y_resampled == 1])
    }

    # Data After Train-Test Split
    data_after_split = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_legitimate": len(y_train[y_train == 0]),
        "train_fraudulent": len(y_train[y_train == 1]),
        "test_legitimate": len(y_test[y_test == 0]),
        "test_fraudulent": len(y_test[y_test == 1])
    }

    return render_template(
        'preprocessing.html',
        dataset_info=dataset_info,
        missing_values=missing_values,
        missing_values_check=missing_values_check,
        duplicates_count_before=duplicates_count_before,  # Pass duplicates count before removal
        duplicates_count_after=duplicates_count_after,  # Pass duplicates count after removal
        duplicates_check=duplicates_check,
        data_after_duplicates=data_after_duplicates,
        X_before_scaler=X_before_scaler.head(10).to_dict(orient='records'),  # Pass as list of dictionaries
        X_after_scaler=X_after_scaler,  # Pass as list of dictionaries
        class_dist_before_smote=class_dist_before_smote,
        class_dist_after_smote=class_dist_after_smote,
        data_before_split=data_before_split,
        data_after_split=data_after_split
    )
@app.route('/comparison')
def comparison():
    # Ensure the results dictionary is populated
    if not results:
        return "No results available. Please train models first.", 400

    # Generate a bar chart comparing all algorithms
    algorithms = list(results.keys())
    accuracy = [results[model]['accuracy'] for model in algorithms]
    precision = [results[model]['classification_report']['1']['precision'] for model in algorithms]
    recall = [results[model]['classification_report']['1']['recall'] for model in algorithms]
    f1_score = [results[model]['classification_report']['1']['f1-score'] for model in algorithms]

    x = np.arange(len(algorithms))  # the label locations
    width = 0.2  # the width of the bars

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, accuracy, width, label='Accuracy')
    plt.bar(x - 0.5 * width, precision, width, label='Precision')
    plt.bar(x + 0.5 * width, recall, width, label='Recall')
    plt.bar(x + 1.5 * width, f1_score, width, label='F1-Score')

    plt.xlabel('Algorithm')
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics Comparison')
    plt.xticks(x, algorithms)
    plt.legend()
    plt.ylim(0, 1)  # Metrics are between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    comparison_plot = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template('comparison.html', results=results, comparison_plot=comparison_plot)

if __name__ == '__main__':
    app.run(debug=True)