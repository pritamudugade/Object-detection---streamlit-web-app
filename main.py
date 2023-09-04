import streamlit as st
import glob
from PIL import Image
import torch
import cv2
import os
import time

st.set_page_config(layout="wide")

# Specify the model path here
cfg_model_path = 'models/yolov5s.pt'

# Check if the model file exists
if not os.path.isfile(cfg_model_path):
    st.error("Model file not found at the specified path. Please ensure the model file exists.")
    st.stop()

model = None
confidence = .25

# Author details
st.sidebar.markdown("Author: MobiNext Technologies")
st.sidebar.markdown("Task: Real-time object detection")

def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        if img_path:
            img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
            if 1 <= img_slider <= len(img_path):
                img_file = img_path[img_slider - 1]
            else:
                st.error("Invalid image selection.")
        else:
            st.error("")

    # Center-align the widgets
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    if data_src != 'Sample data':
        with col1:
            st.markdown("## Height")
            height = st.number_input("Height", min_value=120, step=20, value=720)

        with col2:
            st.markdown("## Width")
            width = st.number_input("Width", min_value=120, step=20, value=1280)

        with col3:
            st.markdown("## FPS")
            fps = st.number_input("FPS", min_value=1, step=1, value=30)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file, height, width, fps)
            st.image(img, caption="Model prediction")

def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"

    # Center-align the widgets
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("## Height")
        height = st.number_input("Height", min_value=120, step=20, value=720)

    with col2:
        st.markdown("## Width")
        width = st.number_input("Width", min_value=120, step=20, value=1280)

    with col3:
        st.markdown("## FPS")
        fps = st.number_input("FPS", min_value=1, step=1, value=30)

    if vid_file:
        cap = cv2.VideoCapture(vid_file)

        st1, st2 = st.columns(2)
        with st1:
            st.image("", caption="Video Input")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0

        frame_skip = 5  # Adjust frame skipping as needed

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame...")
                break

            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img, objects = infer_image(frame, height, width, fps)
            output.image(output_img, caption="Model prediction")

def infer_image(img, height, width, fps):
    model.conf = confidence
    result = model(img, size=(height, width), fps=fps)
    result.render()
    image = Image.fromarray(result.ims[0])
    
    return image

@st.cache_resource
def load_model(path):
    try:
        model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
        return model_
    except Exception as e:
        st.error(f"Error loading the YOLOv5 model: {str(e)}")
        return None

def main():
    global model, confidence, cfg_model_path

    # Center-align the title
    st.markdown("<h1 style='text-align: center;'>VAMS-MobiNext</h1>", unsafe_allow_html=True)

    st.sidebar.title("Custom settings")

    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please add it to the model folder.", icon="⚠️")
    else:
        model = load_model(cfg_model_path)
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)

    st.sidebar.markdown("---")

    input_option = st.sidebar.radio("Select input type: ", ['Image', 'Video'])
    data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload data from local system'])

    if input_option == 'Image':
        image_input(data_src)
    else:
        video_input(data_src)

if __name__ == "__main__":
    main()
