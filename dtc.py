import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Diabetes_PIMA.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details below to predict whether the person has diabetes or not.")

# User Details
name = st.text_input("Patient Name")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5,
    format="%.3f"
)
age = st.number_input("Age", min_value=1, max_value=120, value=25)

# Predict Button
if st.button("Predict Diabetes"):

    user_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    prediction = model.predict(user_data)[0]

    st.subheader(f"Patient: {name}")

    if prediction == 1:
        st.error("⚠️ High Risk: Diabetes Detected")
    else:
        st.success("✅ No Diabetes Detected")

    # Probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(user_data)
        st.write(f"Probability of Diabetes: **{prob[0][1]*100:.2f}%**")
