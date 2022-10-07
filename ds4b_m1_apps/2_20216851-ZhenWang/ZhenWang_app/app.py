import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import umap.umap_ as umap
import seaborn as sns
sns.set()
import streamlit as st
import pickle
import itertools
import shap
from streamlit_shap import st_shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="HR Analysis",
    page_icon="🔍")

hru=pd.read_csv("hru.csv")

st.title('Predict Your Income in Company X')

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.image('https://images.unsplash.com/photo-1633158829585-23ba8f7c8caf?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2340&q=80', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# ---- READ PREPROCESSED MODEL ----
@st.experimental_singleton
def read_objects():
    model_xgb = pickle.load(open('model_xgb.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    ohe_pkl = pickle.load(open('ohe.pkl','rb'))
    shap_values = pickle.load(open('shap_values.pkl','rb'))
    cats = list(itertools.chain(*ohe_pkl.categories_))
    print(len(cats))
    return model_xgb, scaler, ohe_pkl, cats, shap_values

model_xgb, scaler, ohe_pkl, cats, shap_values = read_objects()


# ---- DEFINE EXPLAINER ----
explainer = shap.TreeExplainer(model_xgb)

# ---- COLLECT INPUT FROM USER ----
job_role=st.selectbox('Expected Job Position',options=ohe_pkl.categories_[2])
age=st.number_input('Age',0,100)
gender=st.selectbox('Gender',options=ohe_pkl.categories_[1])
marital=st.selectbox('Marital Situation',options=ohe_pkl.categories_[3])
edu=st.select_slider('Education Level',[1,2,3,4,5])
with st.expander("What's Education Level?"):
    st.markdown("""
    1-Below College / 2-College / 3-Bachelor / 4-Master / 5-Doctor
    """)
dfh=st.number_input('Distance From Home to Company (KM)',0,500)
num_companiesworked=st.number_input('Total Number of Companies You Have Been Worked',0,100)
years_working=st.number_input('Years of Working Before',0,100)
stock_option=st.select_slider('Expected Stock Option Level',[1,2,3,4,5])
training=st.number_input('Expected Training Times per year',0,100)

performance=st.select_slider('Expected Job Performance Level',[1,2,3,4,5])
job_involvement=st.select_slider('Expected Job Involvement Level',[1,2,3,4,5])
satisfaction_job=st.select_slider('Expected Job Satisfaction Level',[1,2,3,4,5])
wlb=st.select_slider('Expected Work Life Balance Level',[1,2,3,4,5])
business_travel=st.selectbox('Expected Business Travel Frequency',options=ohe_pkl.categories_[0])



# make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
if st.button('Calaulate'):
            # make a DF for the numericals and standard scale
            new_df_num = pd.DataFrame({'age':age,
                               'dfh':dfh,
                               'edu':edu,
                               'stock_option':stock_option,
                               'num_companiesworked':num_companiesworked,
                               'training':training,
                               'years_working':years_working,
                               'performance':performance,
                               'job_involvement':job_involvement,
                               'satisfaction_job':satisfaction_job,
                               'wlb':wlb}, index=[0])
            new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns =new_df_num.columns, index=[0])  
            
            new_df_cat = pd.DataFrame({'business_travel':business_travel,
                               'gender':gender,
                               'job_role':job_role,
                               'marital':marital}, index=[0])
            new_values_cat = pd.DataFrame(columns=['Non-Travel','Travel_Frequently','Travel_Rarely','Female','Male','Healthcare Representative','Human Resources','Laboratory Technician','Manager','Manufacturing Director','Research Director','Research Scientist','Sales Executive','Sales Representative','Divorced','Married','Single'],dtype='object')
            new_values_cat['Non-Travel'] = 0
            new_values_cat['Travel_Frequently'] = 0
            new_values_cat['Travel_Rarely'] = 0
            new_values_cat['Female'] = 0
            new_values_cat['Male'] = 0
            new_values_cat['Healthcare Representative'] = 0
            new_values_cat['Human Resources'] = 0
            new_values_cat['Laboratory Technician'] = 0
            new_values_cat['Manager'] = 0
            new_values_cat['Manufacturing Director'] = 0
            new_values_cat['Research Director'] = 0
            new_values_cat['Research Scientist'] = 0
            new_values_cat['Sales Executive'] = 0
            new_values_cat['Sales Representative'] = 0
            new_values_cat['Divorced'] = 0
            new_values_cat['Married'] = 0
            new_values_cat['Single'] = 0
            new_values_cat[business_travel] = 1
            new_values_cat[gender] = 1
            new_values_cat[job_role] = 1
            new_values_cat[marital] = 1

            line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)
            predicted_value = model_xgb.predict(line_to_pred)[0]
            st.metric(label="Predicted Income", value=f'{round(predicted_value)} Ruppes per month' )
            
            SOCIAL_MEDIA = {"Check Today's Exchange Rate": "https://www.xe.com/currencyconverter/"}
            st.write('\n')
            cols = st.columns(len(SOCIAL_MEDIA))
            for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
                cols[index].write(f"[{platform}]({link})")
            
            with st.expander("Explaination for Prediction"):
                st.markdown("""
                The outcome is calculated on the basis of supervised machine learning technology, predicting potential income based on the current employees' feature data and your input. The following chart shows the rationale of specific features. This model will be continuously optimized to reduce MSE in the future.
                """)
            shap_value = explainer.shap_values(line_to_pred)
            st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=500)
            
st.subheader(f'Heartbeat?  Create Digital Resume RIGHT NOW')

# ---- LINK TO RESUME ----
if "my_input" not in st.session_state:
    st.session_state["my_input"]=""
if "my_input_1" not in st.session_state:
    st.session_state["my_input_1"]=""
if "my_input_2" not in st.session_state:
    st.session_state["my_input_2"]=""
if "my_input_3" not in st.session_state:
    st.session_state["my_input_3"]=""
if "my_input_4" not in st.session_state:
    st.session_state["my_input_4"]=""
if "my_input_5" not in st.session_state:
    st.session_state["my_input_5"]=""
if "my_input_6" not in st.session_state:
    st.session_state["my_input_6"]=""
if "my_input_7" not in st.session_state:
    st.session_state["my_input_7"]=""
if "my_input_8" not in st.session_state:
    st.session_state["my_input_8"]=""
if "my_input_9" not in st.session_state:
    st.session_state["my_input_9"]=""
if "my_input_10" not in st.session_state:
    st.session_state["my_input_10"]=""
if "my_input_11" not in st.session_state:
    st.session_state["my_input_11"]=""
if "my_input_12" not in st.session_state:
    st.session_state["my_input_12"]=""
if "my_input_13" not in st.session_state:
    st.session_state["my_input_13"]=""
if "my_input_14" not in st.session_state:
    st.session_state["my_input_14"]=""

my_input = st.text_input("👤 Name", st.session_state["my_input"])
my_input_1 = st.text_input("📖 Discription", st.session_state["my_input_1"])
my_input_2 = st.text_input("📮 Email", st.session_state["my_input_2"])
my_input_3 = st.text_input("🏆 Project Experience", st.session_state["my_input_3"])
with st.expander("Tips for experience writing"):
    st.markdown("""
    Remember to use short sentence and try to use STAR (situation-target-ation-result) to describe your previous experience.
    """)
my_input_4 = st.text_input("💼 Most Recent Work Experience (1/2): Job Title", st.session_state["my_input_4"])
my_input_5 = st.text_input("🕙 Most Recent Work Experience (1/2): Time Duration", st.session_state["my_input_5"])
my_input_6 = st.text_input("✨ Most Recent Work Experience (1/2): Key Outcome (1/2)", st.session_state["my_input_6"])
my_input_7 = st.text_input("✨ Most Recent Work Experience (1/2): Key Outcome (2/2)", st.session_state["my_input_7"])
my_input_8 = st.text_input("💼 Most Recent Work Experience (2/2): Job Title", st.session_state["my_input_8"])
my_input_9 = st.text_input("🕙 Most Recent Work Experience (2/2): Time Duration", st.session_state["my_input_9"])
my_input_10 = st.text_input("✨ Most Recent Work Experience (2/2): Key Outcome (1/2)", st.session_state["my_input_10"])
my_input_11 = st.text_input("✨ Most Recent Work Experience (2/2): Key Outcome (2/2)", st.session_state["my_input_11"])
my_input_12 = st.text_input("📝 Skills (1/3)", st.session_state["my_input_12"])
my_input_13 = st.text_input("📝 Skills (2/3)", st.session_state["my_input_13"])
my_input_14 = st.text_input("📝 Skills (3/3)", st.session_state["my_input_14"])


submit=st.button("create digital resume")
if submit:
    st.session_state["my_input"]=my_input
    st.session_state["my_input_1"]=my_input_1
    st.session_state["my_input_2"]=my_input_2
    st.session_state["my_input_3"]=my_input_3
    st.session_state["my_input_4"]=my_input_4
    st.session_state["my_input_5"]=my_input_5
    st.session_state["my_input_6"]=my_input_6
    st.session_state["my_input_7"]=my_input_7
    st.session_state["my_input_8"]=my_input_8
    st.session_state["my_input_9"]=my_input_9
    st.session_state["my_input_10"]=my_input_10
    st.session_state["my_input_11"]=my_input_11
    st.session_state["my_input_12"]=my_input_12
    st.session_state["my_input_13"]=my_input_13
    st.session_state["my_input_14"]=my_input_14











