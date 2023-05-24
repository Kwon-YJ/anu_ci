import streamlit as st
from datetime import time, datetime

st.header("st.slider")

# ex 1

st.subheader("Slider")

age = st.slider("How old are you?", 0, 130, 25)

st.write(f"I'm {age} years old")

# ex 2

st.subheader("Range slider")

values = st.slider(
    "Select a range of values",
    0.0, 100.0, (25.0, 75.0))
st.write("Values:", values)

# ex 3

st.subheader("Range time slider")

appointment = st.slider(
    "Schedule your appointment:",
    value=(time(11,30), time(12,45)))
st.write(f"You're scheduled for:{appointment}")

# ex 4

st.subheader("Datetime slider")

start_time = st.slider(
    "When do you start?",
    value = datetime(2020, 1, 1, 9, 30),
    format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)
