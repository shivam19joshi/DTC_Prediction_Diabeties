import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Diabetes Risk Analyzer",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df = pd.read_csv("Diabetes_PIMA.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🩺 Diabetes Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset Analytics", "Prediction"]
)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
if menu == "Home":

    st.title("🩺 Smart Diabetes Risk Analyzer")

    st.markdown("""
    This dashboard predicts whether a patient is likely to have diabetes
    using Machine Learning and provides dataset analytics.
    """)

    total_patients = len(df)
    diabetic = len(df[df["Outcome"] == 1])
    non_diabetic = len(df[df["Outcome"] == 0])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Patients",
        total_patients
    )

    col2.metric(
        "Diabetic Patients",
        diabetic
    )

    col3.metric(
        "Non-Diabetic Patients",
        non_diabetic
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        df["Outcome"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            labels=["Non-Diabetic","Diabetic"],
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

# ---------------------------------------------------
# ANALYTICS PAGE
# ---------------------------------------------------
elif menu == "Dataset Analytics":

    st.title("📊 Dataset Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Outcome Distribution")
        fig, ax = plt.subplots()
        sns.countplot(
            data=df,
            x="Outcome",
            ax=ax
        )
        st.pyplot(fig)

    with col2:
        st.subheader("Glucose Distribution")
        fig, ax = plt.subplots()
        sns.histplot(
            df["Glucose"],
            bins=20,
            kde=True,
            ax=ax
        )
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("BMI Distribution")
        fig, ax = plt.subplots()
        sns.histplot(
            df["BMI"],
            bins=20,
            kde=True,
            ax=ax
        )
        st.pyplot(fig)

    with col4:
        st.subheader("Age vs Glucose")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="Age",
            y="Glucose",
            hue="Outcome",
            ax=ax
        )
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------
else:

    st.title("🔍 Diabetes Prediction")

    st.write("Enter patient details below")

    col1, col2 = st.columns(2)

    with col1:

        name = st.text_input(
            "Patient Name"
        )

        weight = st.number_input(
            "Weight (kg)",
            30,
            200,
            70
        )

        height = st.number_input(
            "Height (cm)",
            100,
            220,
            170
        )

        age = st.number_input(
            "Age",
            1,
            100,
            30
        )

        pregnancies = st.number_input(
            "Pregnancies",
            0,
            20,
            0
        )

    with col2:

        glucose = st.number_input(
            "Glucose",
            0,
            300,
            120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            0,
            200,
            70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            0,
            100,
            20
        )

        insulin = st.number_input(
            "Insulin",
            0,
            900,
            80
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            0.000,
            3.000,
            0.500
        )

    # BMI
    bmi = weight / ((height / 100) ** 2)

    st.metric(
        "Calculated BMI",
        round(bmi, 2)
    )

    if bmi < 18.5:
        st.warning("Underweight")
    elif bmi < 25:
        st.success("Normal Weight")
    elif bmi < 30:
        st.warning("Overweight")
    else:
        st.error("Obese")

    st.markdown("---")

    if st.button("Predict Diabetes"):

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=X.columns
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader(f"Patient: {name}")

        st.progress(int(probability * 100))

        st.metric(
            "Diabetes Risk",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:

            st.error(
                "⚠️ High Risk: Diabetes Detected"
            )

        else:

            st.success(
                "✅ Low Risk: No Diabetes Detected"
            )

        st.markdown("---")

        result_df = pd.DataFrame({
            "Parameter": [
                "Age",
                "Weight",
                "Height",
                "BMI",
                "Glucose",
                "Blood Pressure"
            ],
            "Value": [
                age,
                weight,
                height,
                round(bmi,2),
                glucose,
                blood_pressure
            ]
        })

        st.subheader("Patient Summary")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        csv = result_df.to_csv(index=False)

        st.download_button(
            "📥 Download Report",
            csv,
            "patient_report.csv",
            "text/csv"
        )
