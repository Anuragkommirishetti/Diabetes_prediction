import pickle
import pandas as pd
import os
import streamlit as st

model = pickle.load(open("notebook/LogisticModel.pkl","rb"))
scalar = pickle.load(open("notebook/scaler.pkl","rb"))

st.title("Diabetes Predictor")

name = st.text_input("Enter Your Name")
pregnance = st.slider("Number of Pregnancies",0,17)
age = st.slider("Age",18,100)
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skinthickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")


if st.button("Submit"):
    data_df = pd.DataFrame([[pregnance,glucose,bp,skinthickness,insulin,bmi,dpf,age]], 
                       columns=['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age'])
    data = scalar.transform(data_df)
    output = model.predict(data)

    if output[0] == 1:
        st.markdown(
            f"""
            <style>
            @keyframes discoGlow {{
                0% {{ border-color: red; box-shadow: 0 0 10px red; }}
                25% {{ border-color: yellow; box-shadow: 0 0 10px yellow; }}
                50% {{ border-color: green; box-shadow: 0 0 10px green; }}
                75% {{ border-color: blue; box-shadow: 0 0 10px blue; }}
                100% {{ border-color: red; box-shadow: 0 0 10px red; }}
            }}
            .disco-text {{
                font-size: 2em;
                font-weight: bold;
                text-align: center;
                border: 5px solid;
                border-radius: 10px;
                padding: 10px;
                animation: discoGlow 2s infinite;
            }}
            </style>
            <div class="disco-text">{name} is Diabetic. Please consult a doctor for timely advice and treatment.</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <style>
            @keyframes discoGlow {{
                0% {{ border-color: lime; box-shadow: 0 0 10px lime; }}
                25% {{ border-color: cyan; box-shadow: 0 0 10px cyan; }}
                50% {{ border-color: magenta; box-shadow: 0 0 10px magenta; }}
                75% {{ border-color: orange; box-shadow: 0 0 10px orange; }}
                100% {{ border-color: lime; box-shadow: 0 0 10px lime; }}
            }}
            .disco-text {{
                font-size: 2em;
                font-weight: bold;
                text-align: center;
                border: 5px solid;
                border-radius: 10px;
                padding: 10px;
                animation: discoGlow 2s infinite;
            }}
            </style>
            <div class="disco-text">No diabetes detected for {name}. Maintain a healthy lifestyle to stay well.</div>
            """,
            unsafe_allow_html=True
        )
        

