import streamlit as st

st.header("st.selectbox")

option = st.selectbox(
    "what is your favorite colour?",
    ("Blue", "Red", "Green"))

st.write("Your favorite color is ", option)

