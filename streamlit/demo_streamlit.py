import os
import time

# streamlit
import streamlit as st




# img_list = os.listdir("GT_old")
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
    while 1:
        
        image_list = os.listdir("streamlit_save_dir")


        print(len(image_list))

        if len(image_list) != 0:
            image_list = [i[:-4] for i in image_list]
            image_list = list( map(float, image_list) )
            img = str(min(image_list))

            placeholder.image(f"streamlit_save_dir/{img}.jpg", width=600)

            if len(image_list) > 3:
                os.remove(f'streamlit_save_dir//{img}.jpg')
            else:
                time.sleep(0.023)
                continue



if st.sidebar.button("camera off"):
    placeholder.empty()















