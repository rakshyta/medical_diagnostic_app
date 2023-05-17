import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title('Medical Diagnostic Prediction App⚕️')
st.markdown('Does the person has Diabetes ?')

# step 1: load the trained model

model = open('rfc.pickle', 'rb')
clf = pickle.load(model)
model.close()


# step 2: get the user input from front end - create widgets

pregs = st.number_input('Pregnancies', 0, 20, 1)
glucose = st.slider('Glucose', 40, 200, 40)
bp = st.slider('BloodPressure', 20, 140, 20)
skin = st.slider('SkinThickness', 7, 99, 7)
insulin = st.slider('Insulin', 14, 850, 14)
bmi = st.slider('BMI', 18, 70, 18)
dpf = st.slider('DiabetesPedigreeFunction', 0.05, 2.5, 0.05)
age = st.slider('Age', 21, 90, 21)

    
# step 3: convert user input to model input

data = {'Pregnancies':pregs,
        'Glucose':glucose,
        'BloodPressure':bp,
        'SkinThickness':skin,
        'Insulin':insulin,
        'BMI':bmi,
        'DiabetesPedigreeFunction':dpf,
        'Age':age}

input_data = pd.DataFrame([data])


# step 4: get the predictions and print the results

predictions = clf.predict(input_data)[0]
if st.button('Predict'):
    if predictions == 0:
        st.write('The person is Healthy')
    if predictions == 1:
        st.write('The person has Diabetes')
