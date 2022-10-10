from email.policy import default
from pickletools import float8
import streamlit as st
from streamlit.components.v1 import html
import altair as alt
import pandas as pd
from numpy import NaN
import matplotlib.pyplot as plt
import datetime
import requests
import pickle # un-pickling stuff from training notebook
import itertools # we need that to flatten ohe.categories_ into one list for columns
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Bank marketing prediction",
    page_icon="💸",
    initial_sidebar_state="expanded"
)

# RASMUS DATA
@st.experimental_singleton
def load_data():
    bank_data = pd.read_csv("./bank_marketing.csv", sep=';')
    bank_data.drop('default', axis=1, inplace =True)
    bank_data = bank_data.replace('unknown', NaN)
    bank_data_clean = bank_data.dropna(subset=['job', 'marital', 'education', 'housing', 'loan'])
    return bank_data_clean
bank_data_clean = load_data()


# IMPORT PICKLE FILES 
@st.experimental_singleton #decorator and 0-parameters function to only load and preprocess once
def read_objects():
    model_xgb = pickle.load(open('pickles/final_model.pkl','rb'))
    scaler = pickle.load(open('pickles/scalersml.pkl','rb'))
    ohe = pickle.load(open('pickles/ohe.pkl','rb'))
    cats = list(itertools.chain(*ohe.categories_))
    return model_xgb, scaler, ohe, cats

model_xgb, scaler, ohe, cats = read_objects()


# ["Predict customer", "Data inspection"]

with st.sidebar:
    html('''<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap" rel="stylesheet">
<style>
    h2 {
        color: black;
        animation: changeColor 4s linear alternate;
        animation-iteration-count: infinite;
    }
    @keyframes changeColor {
        to {
            color: grey;
        }
    }
</style>
<h2 style="font-size:2rem">BANQUE SAVOIR</h2>
<p style="color:#262730;font-family: Source Sans Pro, sans-serif;font-size:1rem;margin-top:-1rem;">Please enter your customer\'s details</p>
''', height=95)

    col5, col6 = st.columns(2)
    with col5:
        html('''\
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap" rel="stylesheet">
<span style="color:#262730;font-family: Source Sans Pro, sans-serif;line-height:100vh;">Select age:</span>    
        ''', height=86)
    with col6:
        age = st.number_input('', min_value=18, max_value=99)
    marital_type = st.radio(
        "Choose marital status",
        ["Married", "Divorced", "Single"],
        horizontal=True
        )
    col3, col4 = st.columns(2)
    with col3:
        html('''\
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap" rel="stylesheet">
        <span style="color:#262730;font-family: Source Sans Pro, sans-serif;line-height:20vh;position:absolute;top:50%;transform:translateY(-50%)">Select an education level:</span>    
        ''', height=100)
    with col4:
        education = st.selectbox(
            '', 
            options=['Basic of 4 years', 'High school', 'Basic of 6 years', 'Basic of 9 years',
            'Professional course', 'University degree', 'Illiterate']
        )
    contact = st.radio(
        "Select how the customer will be contacted",
        ("Telephone", "Cellular"),
        horizontal=True
        )
    col1, col2 = st.columns(2)
    with col1:
        html('''\
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap" rel="stylesheet">
<span style="color:#262730;font-family: Source Sans Pro, sans-serif;line-height:100vh">Select a job</span>    
        ''', height=86)
    with col2:
        job = st.selectbox('', options=['housemaid', 'services', '👨‍💼 admin.', '👨‍🔧 technician', '👨‍🏭 blue-collar',
        'retired', '💼 management', 'unemployed', 'self-employed',
        '🚀 entrepreneur', '🎓 student'])
        
        poutcome = st.radio(
        "How was the previous campaign for that client",
        ('nonexistent', 'failure', 'success'),
        horizontal=True
        )    

    day = st.selectbox('Select the day', options=['mon', 'tue', 'wed.', 'thu', 'fri'])
    month = st.selectbox('Select the month', options=['mar','apr','may', 'jun', 'jul', 'aug', 'sep','oct','nov','dec'])

    with st.expander("📊 External factors"):
        confidence = st.number_input("Confidence", step=0.1, min_value=-36.4, max_value=100.0)
        euribor3m = st.number_input("Euribor3m", min_value=4.857, max_value=100.0)
        employed = st.number_input("Employed", min_value=5191, max_value=10000)
        duration = st.number_input("Duration", min_value=307, max_value=1000)
        campaign = st.number_input("Campaign", min_value=1, max_value=43)
        previous = st.number_input("Previous", min_value=1, max_value=7)
        variable = st.number_input("Variable", min_value=1.1, max_value=5.0)
        price = st.number_input("Price", min_value=93.994, max_value=2000.0)
        housing_bool = st.radio("Housing", [0, 1], horizontal=True)
        loan_bool = st.radio("Loan", [0, 1], horizontal=True)

    html('''\
<head>
    <style>
        span {
            font-weight: bolder;
        }
        .box {
            width: 2rem;
            height: 2rem;
            background-color: white;
            position: relative;
            cursor: help;
            border: 2px solid #A68101;
        }
        span {
            position: absolute;
            top:50%;
            left:50%;
            transform: translate(-50%, -50%);
        }
        .box:hover {
            animation: turn 2s ease-in-out alternate;
            animation-iteration-count: 2;
        }
        @keyframes turn {
            to {
                transform: rotate(360deg);
            }
        }
        .container {
            position: relative;
            width: fit-content;
            height: fit-content;
            transition: 1s transform;
        }
        .container:hover::before {
            transform: scale(1);
            cursor: help;
        }
        .container::before{
            transform: scale(0);
            content: "How about you? Enter your stats as well!";
            width: 10rem;
            padding: 1rem;
            border-radius: 5px;
            height: fit-content;
            background: white;
            display: block;
            position: absolute;
            left: 120%;
            font-family: Source Sans Pro, sans-serif;
            font-size: 14px;
        }
    </style>
</head>
<div class="container">
    <div class="box">
        <span>?</span>
    </div>
</div>
    ''', height=100)
        

    # MAIN CONTENT

    # make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
if st.button('Predict!'):
    # make a DF for categories and transform with one-hot-encoder
    new_df_cat = pd.DataFrame({'marital_type': marital_type,
                'education':education, 'contact':contact, 'job':job, 'month':['nov'], 
                'day_of_week':['Thu'], 'poutcome':poutcome}, index=[0])
    new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = cats, index=[0])

    # make a DF for the numericals and standard scale
    new_df_num = pd.DataFrame({'age': age, 'cons.conf.idx' :confidence, 
                'euribor3m': euribor3m, 'nr.employed':employed, 
                'emp.var.rate':variable, 'cons.price.idx':price, 
                'campaign':campaign, 'previous' :previous}, index=[0])  
    new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
    
    # make a DF for the boolean values which were converting into booleans
    new_df_bool = pd.DataFrame({'housing_bool':housing_bool,'loan_bool':loan_bool}, columns = ['housing_bool','loan_bool'], index =[0]) 

    #bring all columns together
    line_to_pred = pd.concat([new_values_num, new_values_cat, new_df_bool], axis=1)

    #run prediction for 1 new observation
    value = model_xgb.predict(line_to_pred)[0]

    predicted_value = st.metric(label="Predicted sucess rate", value=f'{round(value,2)} %')

    #print out result to user
    #st.metric(label="Predicted sucess rate", value=f'{round(predicted_value,2)} %')

    c1, c2 = st.columns(2)
    with c2:
        date_input = st.date_input("Selected date:", datetime.datetime.now())
        if date_input.strftime("%B") == "January" or date_input.strftime("%B") == "February" and date_input.strftime("%w") == "0" or date_input.strftime("%w") == "6":
            st.error("Please select any day, but Saturday or Sunday as well a month other than January and February.")
        elif date_input.strftime("%B") == "January" or date_input.strftime("%B") == "February":
            st.error("Please select any month, but January or February.")
        elif date_input.strftime("%w") == "0" or date_input.strftime("%w") == "6":
            st.error("Please select any day, but Saturday or Sunday.")

    st.markdown("### Your chances in customer retention")

    column_1, column_2 = st.columns(2)



    # MODEL SCORE HARD CODED HERE
    model_score = 0.95
    # ACCURACY HARD CODED HERE
    predicted_value = model_xgb.predict(line_to_pred)[0]
    pred_to_100 = round(predicted_value *100,2)
    unsuccessful = 100 - pred_to_100

    with column_1:
        source = pd.DataFrame({"category": ["Success (%)", "Unsuccessful (%)"], "value": [pred_to_100, unsuccessful]})
        base = alt.Chart(source).encode(
            theta=alt.Theta("value:Q", stack=True), color=alt.Color("category:N", scale=alt.Scale(scheme='yellowgreen'))
        )
        pie = base.mark_arc(outerRadius=100)
        text = base.mark_text(radius=130, size=16).encode(text="value:N")
        altair_chart = pie + text
        st.altair_chart(altair_chart, use_container_width=False)

    with column_2:
        highly_likely = '''<span style="font-family: Source Sans Pro, sans-serif;font-size:16px;color:#64B028;font-weight:bolder">highly likely</span>'''
        less_likely = '''<span style="font-family: Source Sans Pro, sans-serif;font-size:16px;color:orange;font-weight:bolder">less likely</span>'''
        not_likely = '''<span style="font-family: Source Sans Pro, sans-serif;font-size:16px;color:red;font-weight:bolder">not likely</span>'''

        accuracy = model_score *100
        
        if predicted_value > 0.8:
            likeliness = highly_likely
        elif predicted_value > 0.5:
            likeliness = less_likely
        elif predicted_value < 0.5:
            likeliness = not_likely

        html(f'''\
        <head>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap" rel="stylesheet">    
        </head>
        <body style="padding: 4rem 0 0 3rem;">
        <p style="font-family: Source Sans Pro, sans-serif;">The person is {likeliness} to convert.</p>
        <p style="font-family: Source Sans Pro, sans-serif;color:rgb(98, 98, 98)">Accuracy: {accuracy}%</p>
        </body>
        ''')

    with st.expander("🔎 Inspect the prediction further"):
        html('''\
    <p style="color:rgb(38, 39, 48);font-family: Source Sans Pro, sans-serif;">Look at the SHAP values to discover why the algorithm chose the result shown above in this case.</p>
        ''', height=50)
        st.markdown("##### HERE WILL BE SHAP VALUE CHART")

    with st.expander("🙋 Feel lost? Get in touch with our customer support"):
        html('''\
    <p style="color:rgb(38, 39, 48);font-family: Source Sans Pro, sans-serif;">In case of questions regarding our software, feel free to reach out to our team <a style="color:#1F78B4" href="mailto:nilsbayer@hotmail.com">via email</a></p>
        ''', height=50)