import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os

# ------------------------------
# STEP 1: ROBUST DATA LOADING
# ------------------------------
@st.cache_data
def load_data():
    # 1. Try relative path (current folder)
    file_name = "student.csv"
    
    # 2. Try the absolute path from your error log
    abs_path = r"C:\Users\S GOKUL KUMAR\OneDrive\Desktop\student.csv"
    
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    elif os.path.exists(abs_path):
        return pd.read_csv(abs_path)
    else:
        return None

# Attempt to load data
df = load_data()

# ------------------------------
# STEP 2: FALLBACK UPLOAD (If file is missing)
# ------------------------------
if df is None:
    st.error("❌ 'student.csv' not found automatically.")
    st.warning("Please upload the file manually below to proceed:")
    uploaded_file = st.file_uploader("Upload student.csv", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()  # Stop the app here until file is uploaded

# ------------------------------
# STEP 3: DATA CLEANING & ENGINEERING
# ------------------------------
# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature Engineering
df['avg_grade'] = (df['G1'] + df['G2'] + df['G3']) / 3
df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# ------------------------------
# STREAMLIT DASHBOARD LAYOUT
# ------------------------------
st.title("🎓 Student Performance Dashboard")
st.success("✔ Data loaded successfully!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Grade Distribution")
    fig1 = px.histogram(df, x="G3", nbins=15, title="Distribution of Final Grades (G3)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📉 Absences vs Grades")
    fig2 = px.scatter(df, x="absences", y="G3", color="studytime", title="Impact of Absences")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# ML MODEL PREPARATION
# ------------------------------
# We remove 'G3', 'pass_fail', and 'avg_grade' from inputs (X) so the model doesn't cheat.
X = df.drop(['pass_fail', 'G3', 'avg_grade'], axis=1)
y_cls = df['pass_fail'] # Target for classification
y_reg = df['G3']        # Target for regression

# Train Classifier
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_c, y_train_c)

# Train Regressor
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X_train_r, y_train_r)

# ------------------------------
# INTERACTIVE PREDICTION SECTION
# ------------------------------
st.markdown("---")
st.subheader("🤖 AI Grade Predictor")
st.write("Enter current student details to predict their final outcome.")

# User Inputs
c1, c2, c3 = st.columns(3)
with c1:
    g1_in = st.number_input("G1 Grade (0-20)", 0, 20, 10)
with c2:
    g2_in = st.number_input("G2 Grade (0-20)", 0, 20, 10)
with c3:
    study_in = st.slider("Study Time (1-4)", 1, 4, 2)

absences_in = st.slider("Absences", 0, 93, 0)

# Prediction Logic
if st.button("Predict Results"):
    # Create input data matching the training columns
    # We take the mean of the dataset to fill in the other columns we didn't ask for (like age, famrel, etc)
    input_data = X.mean().to_frame().T
    
    # Update with user inputs
    input_data['G1'] = g1_in
    input_data['G2'] = g2_in
    input_data['studytime'] = study_in
    input_data['absences'] = absences_in
    
    # Get Predictions
    pred_pass = clf.predict(input_data)[0]
    prob_pass = clf.predict_proba(input_data)[0][1]
    pred_grade = reg.predict(input_data)[0]
    
    # Display Results
    st.write("### Prediction Results:")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if pred_pass == 1:
            st.success(f"PASS/FAIL: **PASS** (Confidence: {prob_pass:.0%})")
        else:
            st.error(f"PASS/FAIL: **FAIL** (Confidence: {1-prob_pass:.0%})")
            
    with col_res2:
        st.info(f"Predicted Final Grade (G3): **{pred_grade:.2f} / 20**")
