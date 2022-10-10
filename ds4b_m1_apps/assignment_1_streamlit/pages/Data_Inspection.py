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
    bank_data = pd.read_csv("../stu_rep_m1/ds4b_m1_apps/assignment_1_streamlit/bank_marketing.csv", sep=';')
    bank_data.drop('default', axis=1, inplace =True)
    bank_data = bank_data.replace('unknown', NaN)
    bank_data_clean = bank_data.dropna(subset=['job', 'marital', 'education', 'housing', 'loan'])
    return bank_data_clean
bank_data_clean = load_data()


# IMPORT PICKLE FILES 
@st.experimental_singleton #decorator and 0-parameters function to only load and preprocess once
def read_objects():
    model_xgb = pickle.load(open('../stu_rep_m1/ds4b_m1_apps/assignment_1_streamlit/pickles/final_model.pkl','rb'))
    scaler = pickle.load(open('../stu_rep_m1/ds4b_m1_apps/assignment_1_streamlit/pickles/scalersml.pkl','rb'))
    ohe = pickle.load(open('../stu_rep_m1/ds4b_m1_apps/assignment_1_streamlit/pickles/ohe.pkl','rb'))
    cats = list(itertools.chain(*ohe.categories_))
    return model_xgb, scaler, ohe, cats

model_xgb, scaler, ohe, cats = read_objects()


# Data Overview
st.header("Data Overview")
col1, col2 = st.columns(2)
col1.metric("# Observations in total", 41.188, "- 7.15%")
col2.metric("Features", 21, "- 3 features")

#Show a table of features
df = pd.read_csv("../stu_rep_m1/ds4b_m1_apps/assignment_1_streamlit/variables_desc.csv")
st.dataframe(data=df) #, width=None, height=None, *, use_container_width=False)

#Show a descriptive table of the numeric features in the cleaned data dataset
st.table(bank_data_clean.describe())




# Graphs
st.markdown('### Explore the succes rate of the campaign based on individual features')

# Selector (plot_bar_succes): column name
selector_plot_bar_succes = st.selectbox(
"Please select the feature you want to explore",
options = bank_data_clean.columns
)

html("", height=10)

# Graph Implementation
def plot_bars_success(column_name):
    #month and deposit
    m_df = pd.DataFrame()

    #Create a column in the previously defined empty dataframe with the yes and no for each month
    m_df['yes'] = bank_data_clean[bank_data_clean['y'] == 'yes'][column_name].value_counts()
    m_df['no'] = bank_data_clean[bank_data_clean['y'] == 'no'][column_name].value_counts()

    #Plot:
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
    #Do the absolute values
    ax1.bar(m_df.index,m_df['yes'], label='yes', alpha=0.5, color='green')
    ax1.bar(m_df.index, m_df['no'], bottom=m_df['yes'], label='no', alpha=0.5, color='orange')
    plt.sca(ax1)
    ax1.set_ylabel("Absolute count")
    ax1.set_xlabel("month")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15,1), ncol=1)

    #Show the Percentage for each month!
    totals = m_df['no'] + m_df['yes']
    data1_rel = m_df['yes'] / totals * 100
    data2_rel = m_df['no'] / totals * 100

    ax2.bar(m_df.index, data1_rel, alpha=0.5, color='green')
    ax2.bar(m_df.index, data2_rel, bottom=data1_rel, alpha=0.5, color='orange')

    ax2.set_ylabel("Percentage")
    ax2.set_xlabel(column_name)
    return f

# Calling the plot
st.pyplot(plot_bars_success(selector_plot_bar_succes))
