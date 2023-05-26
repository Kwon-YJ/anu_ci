import streamlit as st
import pandas as pd

st.title("Input CSV")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("DataFreame")
    st.write(df)
    st.subheader("Descriptive Statistivs")
    st.write(df.describe())
else:
    st.info("☝️ Upload a CSV file")

