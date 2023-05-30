import streamlit as st

st.sidebar.title('Settings')

# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ('YOLOX-s', 'YOLOX-m', 'YOLOX-l', 'YOLOX-l')
)

st.sidebar.button("camera on")

st.title("VESMO(Vespa Monitoring System) 🐝")
st.subheader(f"Powered by {model_type}")

with st.expander("About this app"):
    st.write("This app uses deep learning to detect hornets in real time and stores the result information.")


# mode_type = st. 카메라 on off 버튼

# st.title(f'{model_type} Predictions 🐝')

from random import randint
import time

placeholder = st.empty()

while True:
    time.sleep(2)
    if randint(1, 100) % 2 == 0:
        placeholder.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png', width=250)
    else:
        placeholder.image('https://github.com/Kwon-YJ/binance-trader-c1/blob/master/images/performance2.png?raw=true', width=500)