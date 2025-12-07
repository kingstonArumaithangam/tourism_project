import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Kingston-Personal/tourism-project", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package.
Please provide the customer and interaction details below to get a prediction.
""")

st.header("Customer Details")

# -------------------------
# Categorical Variables
# -------------------------
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])
Designation = st.text_input("Designation", "e.g., Manager, Executive")
ProductPitched = st.selectbox("Product Pitched", ["Wellness Package", "Adventure Package", "Other"])

# -------------------------
# Numerical Variables
# -------------------------
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=20, value=2)
PreferredPropertyStar = st.number_input("Preferred Hotel Star Rating", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=50, value=2)
NumberOfChildrenVisiting = st.number_input("Number of Children Below Age 5", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=4)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=1)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)

# -------------------------
# Prepare Input DataFrame
# -------------------------
# Convert yes/no to 1/0
Passport = 1 if Passport == "Yes" else 0
OwnCar = 1 if OwnCar == "Yes" else 0

input_data = pd.DataFrame([{
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
    "Age": Age,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer Likely to Purchase" if prediction == 1 else "Customer Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"**{result}**")
