import streamlit as st
import time
import os

img_list = os.listdir("GT_old")

st.sidebar.title('Settings')

# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ('YOLOX-s', 'YOLOX-m', 'YOLOX-l', 'YOLOX-l')
)

st.title("Vespa Monitoring System🐝")
st.subheader(f"`Powered by {model_type}`🔋")

with st.expander("About this app"):
    st.write("This app uses deep learning to detect hornets in real time and stores the result information.")

placeholder = st.empty()

if st.sidebar.button("camera on"):
    img_list = list(set(img_list))
    for img in img_list:
        time.sleep(2)
        placeholder.image(f'GT_old//{img}', width=600)
if st.sidebar.button("camera off"):
    placeholder.empty()
