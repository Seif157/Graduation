import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from pathlib import Path

# 1. Load and Prepare Data
file_path = Path(r"E:/Rewad/Final/heart_disease_risk_dataset_earlymed.csv")
df = pd.read_csv(file_path)

# Handle missing values (if any)
df.dropna(inplace=True)

# Define features and target
features = [
    'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
    'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'High_BP',
    'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 'Sedentary_Lifestyle',
    'Family_History', 'Chronic_Stress', 'Gender', 'Age'
]
X = df[features]
y = df['Heart_Risk']

# 2. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 4. Train and Optimize Model
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, scoring='recall', n_jobs=-1)
grid_search.fit(X_train, y_train)
optimized_model = grid_search.best_estimator_

# 5. Evaluate the Model
predictions = optimized_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, optimized_model.predict_proba(X_test)[:, 1])
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

# 6. Save the Model and Scaler
output_dir = Path(r"E:\Rewad\Final\ml_streamlit_app\model")
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "heart_disease_model.pkl"
scaler_path = output_dir / "scaler.pkl"

with open(model_path, 'wb') as model_file:
    pickle.dump(optimized_model, model_file)

with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")
