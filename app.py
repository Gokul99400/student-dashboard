import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AI Grade Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Grade Predictor")
st.write("Enter current student details to predict their final outcome.")

@st.cache_data
def load_data():
    return pd.read_excel("student.xlsx")

data = load_data()

X = data[["G1", "G2", "studytime", "absences"]]
y = data["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

col1, col2, col3 = st.columns(3)

with col1:
    g1 = st.number_input("G1 Grade (0-20)", min_value=0, max_value=20, value=10)

with col2:
    g2 = st.number_input("G2 Grade (0-20)", min_value=0, max_value=20, value=10)

with col3:
    studytime = st.slider("Study Time (1-4)", min_value=1, max_value=4, value=2)

absences = st.slider("Absences", min_value=0, max_value=93, value=5)

if st.button("Predict Results"):
    input_data = np.array([[g1, g2, studytime, absences]])
    predicted_g3 = model.predict(input_data)[0]
    predicted_g3 = max(0, min(predicted_g3, 20))

    PASS_MARK = 10

    if predicted_g3 >= PASS_MARK:
        result = "PASS"
    else:
        result = "FAIL"

    confidence = round((predicted_g3 / 20) * 100, 2)

    st.subheader("Prediction Results")

    colA, colB = st.columns(2)

    with colA:
        if result == "PASS":
            st.success(f"PASS (Confidence: {confidence}%)")
        else:
            st.error(f"FAIL (Confidence: {confidence}%)")

    with colB:
        st.info(f"Predicted Final Grade (G3): {predicted_g3:.2f} / 20")

    if result == "FAIL":
        st.warning(
            "⚠️ The predicted final grade is below the minimum pass mark (10). "
            "Improving internal marks and reducing absences can significantly "
            "increase the final outcome."
        )
    else:
        st.success(
            "✅ The student is likely to pass. "
            "Maintaining consistency in studies will improve the final grade further."
        )
