import streamlit as st

st.header("st.mulitselect")

options = st.multiselect(
    "What are your favortie colors",
    ["Green", "Yellow", "Red", "Blue"],
    ["Yellow", "Red"])

st.write("You selected:", options)
