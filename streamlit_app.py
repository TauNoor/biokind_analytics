import streamlit as st

import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# load model from pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    f.close()

# app title
st.title("Repeat Donor Prediction")

# inputs
avg_donation = st.text_area("Average Donation Amount")
num_donations = st.text_area("Number of Donations")
last_donation = st.text_area("Most Recent Donation")

# Prediction
if st.button("Predict"):
    avg_donation = np.log(
        float(avg_donation)
    )
    num_donations = int(num_donations)
    try:
        last_donation = np.log(
            (pd.Timestamp.today() - pd.to_datetime(last_donation)).days
        )
    except Exception as e:
        st.error("Please input a date")
    
    feature_vector = np.array(
        [avg_donation, num_donations, last_donation]
    ).reshape(1, -1)
    
    prediction = model.predict(feature_vector)[0]
    
    # Returns prediction based on model
    if prediction == 1:
        st.header("Repeat Donor")
    else:
        st.header("Nonrepeat Donor")
        