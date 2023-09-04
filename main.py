import streamlit as st
import glob
import wget
from PIL import Image
import torch
import cv2
import os

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25

# Centered title with HTML and CSS
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <h1>VAMS-MobiNext</h1>
    </div>
    """,
    unsafe_allow_html=True
)

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
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model prediction")

def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        st.markdown("---")
        output = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame...")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img, use_column_width=True)  # Display the video on full screen

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image

@st.cache_resource
def load_model(path):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to('cpu')  # Assume CPU for simplicity
    print("model to CPU")
    return model_

def get_user_model():
    model_src = st.sidebar.radio("Model source", ["Custom model", "YOLO"])
    model_file = None
    if model_src == "Custom model":
        user_model_path = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if user_model_path:
            model_file = "models/uploaded_" + user_model_path.name
            with open(model_file, 'wb') as out:
                out.write(user_model_path.read())
    else:
        url = st.sidebar.text_input("Model URL")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_
    return model_file

def main():
    global model, confidence, cfg_model_path

    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please add it to the model folder.", icon="⚠️")
    else:
        model = load_model(cfg_model_path)
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)

        st.sidebar.markdown("---")

        input_option = st.sidebar.radio("Select input type: ", ['Image', 'Video'])
        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload data from the local system'])

        if input_option == 'Image':
            image_input(data_src)
        else:
            video_input(data_src)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

# Add author details at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)  # Create some space
st.markdown("<p style='text-align: center;'>Created by MobiNext Technologies</p>", unsafe_allow_html=True)
