import streamlit as st
import pandas as pd
import joblib

st.title("Online Shopper Purchase Prediction App")
st.write("""This app predicts whether an online shopper will make a purchase based on their browsing behavior and session characteristics.""")

# Load Model + Selected Features
rf_model = joblib.load("source/t_models/rf_trainined_model.pkl")
rf_features = joblib.load("source/s_features/selected_features_of_rf.pkl")

st.subheader("Please provide visitor browsing information")

# Making sure the users can enter only numeric input, because streamlit requires all values must be 1 datatype.
def numeric_input(label, min_val, max_val, step=1.0):
    return st.number_input(label, min_value=float(min_val), max_value=float(max_val), step=float(step), format="%.3f")

# assigned values for Selected Features of RF model
PageValues = numeric_input("Page Values (Average value of visited pages)", 0.0, 400.0, 0.1)
ProductRelated_Duration = numeric_input("Product Related Duration (seconds)", 0.0, 70000.0, 1.0)
ExitRates = numeric_input("Exit Rate (0.00 - 0.20)", 0.0, 0.20, 0.001)
ProductRelated = int(numeric_input("Product Related Pages Visited", 0, 700, 1))
BounceRates = numeric_input("Bounce Rate (0.00 - 0.20)", 0.0, 0.20, 0.001)
Administrative_Duration = numeric_input("Administrative Duration", 0.0, 5000.0, 1.0)
Administrative = int(numeric_input("Administrative Pages Visited", 0, 30, 1))
Region = int(numeric_input("Region (1 - 9)", 1, 9, 1))
TrafficType = int(numeric_input("Traffic Type (1 - 20)", 1, 20, 1))
Informational_Duration = numeric_input("Informational Duration", 0.0, 3000.0, 1.0)

Month_Nov = st.radio("Is the visit in November?", ["No", "Yes"])
Month_Nov = 1 if Month_Nov == "Yes" else 0

Browser = int(numeric_input("Browser (1–13)", 1, 13, 1))
OperatingSystems = int(numeric_input("Operating System (1–8)", 1, 8, 1))


# After receiving user input, we will do  prediction for it
user_input = {
    "PageValues": PageValues,
    "ProductRelated_Duration": ProductRelated_Duration,
    "ExitRates": ExitRates,
    "ProductRelated": ProductRelated,
    "BounceRates": BounceRates,
    "Administrative_Duration": Administrative_Duration,
    "Administrative": Administrative,
    "Region": Region,
    "TrafficType": TrafficType,
    "Informational_Duration": Informational_Duration,
    "Month_Nov": Month_Nov,
    "Browser": Browser,
    "OperatingSystems": OperatingSystems
}

user_df = pd.DataFrame([user_input])
user_df = user_df.reindex(columns=rf_features, fill_value=0)

# Predicting whether Shoppers will buy product or not.
if st.button("Predict Purchase Intent"):
    pred = rf_model.predict(user_df)[0]
    proba = rf_model.predict_proba(user_df)[0][1]

    if pred == 1:
        st.success(f"Prediction: Shopper WILL purchase (Probability: {proba:.2f})")
    else:
        st.error(f"Prediction: Shopper will NOT purchase (Probability: {proba:.2f})")
