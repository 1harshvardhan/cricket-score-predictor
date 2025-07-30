import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the model pipeline
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline.pkl")

pipe = load_model()

st.title("Cricket Score Predictor 🏏")

# Add "Select Team" and "Select City" as default options
teams = ['Select Team', 'India', 'Australia', 'Sri Lanka', 'New Zealand', 'South Africa', 'England', 'Pakistan', 'Bangladesh']
cities = ['Select City', 'Colombo', 'Dubai', 'Johannesburg', 'Mirpur', 'London', 'Melbourne', 'Sydney', 'Delhi', 'Abu Dhabi', 'Auckland']

# Function to get flag path
def get_flag_path(team):
    filename = team.lower().replace(" ", "_") + ".png"
    return os.path.join("static/images", filename)

# Layout
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select Batting Team", teams, index=0)
    if batting_team != "Select Team":
        st.image(get_flag_path(batting_team), width=80)

with col2:
    # Filter bowling teams to include "Select Team" + others except selected batting team
    bowling_team_options = ['Select Team'] + [t for t in teams if t != batting_team and t != "Select Team"]
    bowling_team = st.selectbox("Select Bowling Team", bowling_team_options, index=0)
    if bowling_team != "Select Team":
        st.image(get_flag_path(bowling_team), width=80)

# City dropdown
city = st.selectbox("Select City", cities, index=0)

# Match details
current_score = st.number_input("Current Score", min_value=0, max_value=300)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
last_five = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100)

# Prediction
if st.button("Predict Final Score"):
    if batting_team == "Select Team" or bowling_team == "Select Team" or city == "Select City":
        st.warning("Please select valid teams and city.")
    elif overs == 0:
        st.warning("Overs must be greater than 0 to calculate run rate.")
    else:
        balls_bowled = int(overs * 6)
        balls_left = 120 - balls_bowled
        wickets_left = 10 - wickets
        current_run_rate = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wicket_left': [wickets_left],
            'current_run_rate': [current_run_rate],
            'last_five': [last_five]
        })

        result = int(pipe.predict(input_df)[0])
        st.subheader(f"🎯 Predicted Final Score: {result} runs")



