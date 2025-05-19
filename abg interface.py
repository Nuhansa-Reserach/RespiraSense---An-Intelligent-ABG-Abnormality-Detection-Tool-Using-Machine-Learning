import streamlit as st
import numpy as np
import joblib

# Load the latest SVM model (compatible with scikit-learn 1.5.1)
model = joblib.load("svm_model_v151.pkl")

# App title
st.title("Blood Gas Analyzer - ABG Abnormality Detection")
st.markdown("📊 This tool uses machine learning to detect abnormalities in arterial blood gas (ABG) results and suggest possible symptoms and next steps.")

# Input section
st.header("Enter ABG Parameters")

pH = st.number_input("pH (Normal: 7.35 - 7.45)", value=7.40, step=0.01)
pCO2 = st.number_input("pCO₂ (Normal: 35 - 45 mmHg)", value=40.0, step=0.1)
pO2 = st.number_input("pO₂ (Normal: 75 - 100 mmHg)", value=90.0, step=0.1)
HCO3 = st.number_input("HCO₃ (Normal: 22 - 26 mEq/L)", value=24.0, step=0.1)
SaO2 = st.number_input("SaO₂ (Normal: 95 - 100%)", value=98.0, step=0.1)

# Predict button
if st.button("Analyze ABG Status"):
    input_data = np.array([[pH, pCO2, pO2, HCO3, SaO2]])
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("✅ Status: Normal")
        st.write("All ABG parameters are within expected physiological ranges.")
    else:
        st.error("⚠️ Status: Abnormal")
        st.markdown("### 🩺 Possible Symptoms:")
        st.markdown("""
        - Shortness of breath  
        - Dizziness or confusion  
        - Rapid breathing  
        - Fatigue or drowsiness  
        - Cyanosis (bluish lips or fingers)
        """)
        st.markdown("### 💡 Suggested Actions:")
        st.markdown("""
        - Repeat ABG test for confirmation  
        - Administer oxygen or ventilation support  
        - Consult a respiratory specialist  
        - Monitor patient for deterioration  
        - Consider ICU referral if symptoms worsen
        """)

# Footer
st.markdown("---")
st.caption("Developed by Nuhansa Herath - Biomedical Engineering Final Year Project")

