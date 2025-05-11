import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
# Assuming your model is in a 'model' directory
MODEL_PATH = os.path.join(os.path.dirname(
    __file__), "model", "heart_disease_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

# Loading the model
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
# Define features
FEATURES = [
    'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
    'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'High_BP',
    'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 'Sedentary_Lifestyle',
    'Family_History', 'Chronic_Stress', 'Gender', 'Age'
]

# App title
st.title("💓 Heart Disease Risk Predictor")

# Sidebar Image
st.sidebar.image(
    'data:image/jpeg;base64,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')

# Create Tabs
select = st.sidebar.selectbox(
    'Choose Option', ['Model Training', 'Risk Prediction'])

# Model Training Tab
if select == 'Model Training':
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)

        st.write("### Data Preview")
        st.write(data.head())

        # Feature and Target Selection
        features = [col for col in data.columns if col != 'Heart_Risk']
        selected_features = st.sidebar.multiselect(
            "Select Input Features", options=features, default=features)
        target = st.sidebar.selectbox(
            "Select Target Variable", options=data.columns)

        # Scaling and Encoding
        if selected_features and target:
            X = data[selected_features]
            y = data[target]

            # Encoding categorical variables
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            # Scaling
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Model Training
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(y_test, y_pred, average='macro') * 100
            recall = recall_score(y_test, y_pred, average='macro') * 100

            # Show Gauges
            st.subheader("Model Performance")
            for metric, value in zip(["Accuracy", "Precision", "Recall"], [accuracy, precision, recall]):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': metric},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "blue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 100], 'color': "blue"}
                        ]
                    }
                ))
                st.plotly_chart(fig)

# Risk Prediction Tab
else:
    st.write("### Risk Prediction")
    user_data = []
    for feature in FEATURES:
        if feature == 'Age':
            value = st.slider(feature, 20, 100, 50)
        elif feature == 'Gender':
            value = st.selectbox(feature, ["Male", "Female"])
            value = 1 if value == "Male" else 0
        else:
            value = st.selectbox(feature, ["No", "Yes"])
            value = 1 if value == "Yes" else 0
        user_data.append(value)

    # Prediction
    user_df = pd.DataFrame([user_data], columns=FEATURES)
    scaled_data = scaler.transform(user_df)
    prediction = model.predict(scaled_data)[0]
    risk = "High Risk" if prediction == 1 else "Low Risk"
    st.success(f"Prediction: {risk}")
