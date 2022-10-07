from pathlib import Path
import streamlit as st
from PIL import Image

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"


# --- GENERAL SETTINGS ---
if "my_input" not in st.session_state:
    st.session_state["my_input"]=""
NAME=st.session_state["my_input"]

if "my_input_1" not in st.session_state:
    st.session_state["my_input_1"]=""
DESCRIPTION=st.session_state["my_input_1"]

if "my_input_2" not in st.session_state:
    st.session_state["my_input_2"]=""
EMAIL=st.session_state["my_input_2"]

if "my_input_3" not in st.session_state:
    st.session_state["my_input_3"]=""
PROJECTS=st.session_state["my_input_3"]

# --- BASIC INFO ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image('https://images.unsplash.com/photo-1517849845537-4d257902454a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=735&q=80',width=200)

with col2:
    st.title(NAME)
    st.write("📒",DESCRIPTION)
    st.write("📫", EMAIL)

# --- PROJECT ---
st.write('\n')
st.subheader("Project Experience")
if "my_input_3" not in st.session_state:
    st.session_state["my_input_3"]=""
st.write("🏆",st.session_state["my_input_3"])


# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Working History (1/2)")
if "my_input_4" not in st.session_state:
    st.session_state["my_input_4"]=""
st.write("💼",st.session_state["my_input_4"])

if "my_input_5" not in st.session_state:
    st.session_state["my_input_5"]=""
st.write(st.session_state["my_input_5"])

if "my_input_6" not in st.session_state:
    st.session_state["my_input_6"]=""
st.write("🌟",st.session_state["my_input_6"])

if "my_input_7" not in st.session_state:
    st.session_state["my_input_7"]=""
st.write("🌟",st.session_state["my_input_7"])

st.write('\n')
st.subheader("Working History (2/2)")
if "my_input_8" not in st.session_state:
    st.session_state["my_input_8"]=""
st.write("💼",st.session_state["my_input_8"])

if "my_input_9" not in st.session_state:
    st.session_state["my_input_9"]=""
st.write(st.session_state["my_input_9"])

if "my_input_10" not in st.session_state:
    st.session_state["my_input_10"]=""
st.write("🌟",st.session_state["my_input_10"])

if "my_input_11" not in st.session_state:
    st.session_state["my_input_11"]=""
st.write("🌟",st.session_state["my_input_11"])

# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
if "my_input_12" not in st.session_state:
    st.session_state["my_input_12"]=""
st.write("🌟",st.session_state["my_input_12"])
if "my_input_13" not in st.session_state:
    st.session_state["my_input_13"]=""
st.write("🌟",st.session_state["my_input_13"])
if "my_input_14" not in st.session_state:
    st.session_state["my_input_14"]=""
st.write("🌟",st.session_state["my_input_14"])

st.button('Submit to HR')
