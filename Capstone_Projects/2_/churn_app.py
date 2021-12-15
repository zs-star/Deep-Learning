import streamlit as st
import pickle
import joblib
import pandas as pd
from PIL import Image
scaler_churn = joblib.load(open("scaler_churn", "rb"))
label_encode_dep = joblib.load(open("label_encode_dep", "rb"))
label_encode_sal = joblib.load(open("label_encode_sal", "rb"))
RF_model=joblib.load(open("RF_model", "rb"))
columns_name=['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'work_accident', 'left',
       'promotion_last_5years', 'departments_', 'salary']
def pred(data):
    data=pd.DataFrame([data])
    #data = data.reindex(columns=columns_name)
    data.departments_=label_encode_dep.transform(data.departments_)
    data.salary=label_encode_sal.transform(data.salary)
    data=scaler_churn.transform(data)
    pred=RF_model.predict(data)
    #return f'the pred {data}'
    if pred==0:
        return f"The employee can STAY"
    else:
        return f"The employee can LEAVE"

        
    

st.sidebar.title('Employee Churn Prediction')
satisfaction=st.sidebar.slider("How satisfied with job?",0.0,1.0,step=0.01)
evaluation=st.sidebar.slider("What is last evaluation",0.0, 1.0, step=0.01)
average_hour=st.sidebar.slider("How many hours in average per month?", 0, 350, step=1)
num_project=st.sidebar.slider("How many project in?", 1, 10, step=1)
num_years=st.sidebar.slider("How many years in Company?", 1, 30, step=1)
work_acc=st.sidebar.radio ("Any work accident?", ('Yes','No'))
promotion=st.sidebar.radio ("Any promotion in last 5 years?", ('Yes','No'))
department=st.sidebar.selectbox("Which department?", ('sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD')) 
salary=st.sidebar.selectbox("What is salary level?", ('low', 'medium', 'high')) 
col1, col2= st.columns([2,1])
with col1:
    img=Image.open('churn.png')
    st.image(img)
my_dict = {
    "satisfaction_level": satisfaction,
    "last_evaluation": evaluation,
    "number_project": num_project,
    "average_montly_hours": average_hour,
    "time_spend_company":num_years,
    "work_accident":work_acc,
    "promotion_last_5years":promotion,
    "departments_":department,
    "salary":salary

}
df1= pd.DataFrame.from_dict([my_dict])
st.table(df1)
if work_acc=='Yes':
    work_acc=1
else:
    work_acc=0  
if  promotion=='Yes':
    promotion=1
else:
    promotion=0  
dict = {
    "satisfaction_level": satisfaction,
    "last_evaluation": evaluation,
    "number_project": num_project,
    "average_montly_hours": average_hour,
    "time_spend_company":num_years,
    "work_accident":work_acc,
    "promotion_last_5years":promotion,
    "departments_":department,
    "salary":salary

}    
#sample=pd.DataFrame.from_dict([dict])
m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #D35400;
            color:#FBFCFC;
        }
        div.stButton > button:hover {
            background-color: #D35400;
            color:#FBFCFC;
            }
        </style>""", unsafe_allow_html=True)
if st.button('Predict'):
            st.write(pred(dict)) 