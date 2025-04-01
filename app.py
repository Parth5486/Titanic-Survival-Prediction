import pickle
import streamlit as st
import numpy as np

# Load the model
with open("logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #003366;'>🚢 Titanic Survival Prediction 🚢</h1>", 
    unsafe_allow_html=True
)
st.write("### Enter Passenger Details Below:")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    pclass = st.radio("🛳 Passenger Class", [1, 2, 3], index=2)
    sex = st.selectbox("⚤ Sex", ["Male", "Female"])
    age = st.slider("🎂 Age", 0, 100, 25)

with col2:
    sibsp = st.number_input("👨‍👩‍👦 Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
    parch = st.number_input("👶 Parents/Children Aboard", min_value=0, max_value=10, step=1)
    fare = st.slider("💰 Fare Paid", 0.0, 500.0, 50.0)

embarked = st.selectbox("📍 Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

# Convert inputs to numeric values
sex_encoded = 1 if sex == "Male" else 0
embarked_encoded = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}[embarked]

features = np.array([[float(pclass), float(sex_encoded), float(age), float(sibsp), 
                      float(parch), float(fare), float(embarked_encoded)]])

# Predict button
if st.button("🚀 Predict"):
    prediction = model.predict(features)  
    result = "🎉 Survived" if prediction[0] == 1 else "⚠ Did Not Survive"
    color = "green" if prediction[0] == 1 else "red"
    
    st.markdown(f"<h2 style='text-align: center; color: {color};'>{result}</h2>", unsafe_allow_html=True)
