import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

def run_model():
    file_path = filedialog.askopenfilename(title="Select a Dataset", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return
    
    try:
        dataset = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")
        return

    missing_values = dataset.isnull().sum()

    algorithm = algorithm_var.get()

    legit_df = dataset[dataset['Class'] == 0].sample(n=450, random_state=42)
    fraud_df = dataset[dataset['Class'] == 1].sample(n=450, random_state=42)

    dataset_combined = pd.concat([legit_df, fraud_df], ignore_index=True)

    X = dataset_combined.drop(columns=['Class'])  
    y = dataset_combined['Class']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = "Logistic Regression:\n"
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = "Decision Tree:\n"
    elif algorithm == "XGBoost":
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = "XGBoost:\n"
    else:
        messagebox.showerror("Error", "No algorithm selected!")
        return

    result += f"Accuracy: {accuracy_score(y_test, y_pred)}\n"
    result += f"Classification Report:\n{classification_report(y_test, y_pred)}"
    
    result_text.delete(1.0, tk.END)  
    result_text.insert(tk.END, result)

window = tk.Tk()
window.title("Fraud Detection Model")

window.geometry("700x600")

algorithm_var = tk.StringVar(window)
algorithm_var.set("Logistic Regression") 

algorithm_label = tk.Label(window, text="Select Algorithm:")
algorithm_label.pack(pady=10)

algorithm_menu = tk.OptionMenu(window, algorithm_var, "Logistic Regression", "Decision Tree", "XGBoost")
algorithm_menu.pack(pady=10)

run_button = tk.Button(window, text="Run", command=run_model)
run_button.pack(pady=20)

result_text = tk.Text(window, height=15, width=80)
result_text.pack(pady=20)

window.mainloop()
