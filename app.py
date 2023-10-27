import streamlit as st
import pickle
import pandas as pd
import sklearn

teams = ['Mumbai Indians', 'Royal Challengers Bangalore',
       'Lucknow Super Giants', 'Kolkata Knight Riders',
       'Chennai Super Kings', 'Rajasthan Royals',
       'Gujarat Titans', 'Sunrisers Hyderabad']

cities = ['Ahmedabad', 'Chennai', 'Mumbai', 'Bengaluru', 'Kolkata', 'Delhi',
       'Hyderabad', 'Lucknow', 'Jaipur']

pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select the host City', sorted(cities))

target = st.number_input('What is the target Score?')

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('What is the current Score')
with col4:
    overs = st.number_input('Which over is going on?')
with col5:
    wickets = st.number_input('What is the number of Wickets?')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")