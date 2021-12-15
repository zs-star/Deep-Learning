import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from PIL import Image
scaler= joblib.load(open("scaler_auto", "rb"))
model_auto = joblib.load(open("final_model_auto", "rb"))
columns_name=['hp_kW', 'km', 'age', 'make_model_Audi A1', 'make_model_Audi A3',
       'make_model_Opel Astra', 'make_model_Opel Corsa',
       'make_model_Opel Insignia', 'make_model_Renault Clio',
       'make_model_Renault Duster', 'make_model_Renault Espace',
       'Gearing_Type_Automatic', 'Gearing_Type_Manual',
       'Gearing_Type_Semi-automatic']
def pred(data):
    data=pd.DataFrame([data])
    data=pd.get_dummies(data)
    data = data.reindex(columns=columns_name, fill_value=0)
    data=scaler.transform(data)
    pred=model_auto.predict(data)
    return f"The predict price: ${pred[0]:.2f}"


st.markdown("<h1 style='text-align: center; color: grey;'>FREE Predict</h1>", unsafe_allow_html=True)
img=Image.open('original.png')
st.image(img)
col1, col2= st.columns([2,1])


with col1:
    hp=st.number_input('Enter horsepower (hp)',min_value=None, max_value=None,step=1)
    age=st.number_input('Enter age',step=1)
    km=st.number_input('Enter kilometre (km)',step=100)
    #model=st.text_input('Enter model')
    model=st.selectbox('Select a  model',['Audi A1', 'Audi A2', 'Audi A3', 'Opel Astra', 'Opel Corsa',
       'Opel Insignia', 'Renault Clio', 'Renault Duster',
       'Renault Espace'])
    #gearing_type=st.text_input('Enter gearing type')   
    gearing_type=st.selectbox('Select a  gearing type',['Automatic', 'Manual', 'Semi-automatic'])
sample={"hp_kW":hp,"age":age,"km":km,"make_model":model,"Gearing_Type":gearing_type}     
with col2:
    st.text("")
    
    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #0099ff;
            color:#ffffff;
        }
        div.stButton > button:hover {
            background-color: #093FE8;
            color:#ffffff;
            }
        </style>""", unsafe_allow_html=True)
    if st.button('Predict'):
            st.write(pred(sample)) 
    st.text("")
    st.text("") 
     
    
    
          

   


