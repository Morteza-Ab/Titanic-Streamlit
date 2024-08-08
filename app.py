import streamlit as st
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgbpipe.joblib')
st.title('Just for fun 😊; if you were in Titanic :ship:')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
passengerid = st.text_input("Input Passenger ID", '8585') 
pclass = st.selectbox("Choose class", [1,2,3])
name  = st.text_input("Input Passenger Name", 'Morteza!!!')
sex = st.select_slider("Choose sex", ['male','female'])
age = st.slider("Choose age",0,100)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,10)
ticket = st.text_input("Input Ticket Number", "8585") 
fare = st.number_input("Input Fare Price", 0,1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.select_slider("Did they Embark?", ['S','C','Q'])

def predict(): 
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Yay! you are still alive :thumbsup:')
    else: 
        st.error('You are not alive anymore 😢') 

trigger = st.button('Predict', on_click=predict)

