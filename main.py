
import streamlit as st
import glob
import wget
from PIL import Image
import torch
import cv2
import os
import time

# Center-align the page content
st.markdown(
    """
    <style>
    .reportview-container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
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
            st.error("please select desired option")
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

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame...")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()




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

def load_custom_model(model_path):
    model = torch.load(model_path, map_location='cpu')  # Assume CPU for simplicity
    model.eval()
    return model

@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file

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

    # Centered title
    st.title("VAMS-MobiNext")

    # Sidebar with custom settings
    st.sidebar.title("Custom settings")
    user_model_path = get_user_model()
    if user_model_path:
        cfg_model_path = user_model_path

    st.sidebar.text(cfg_model_path.split("/")[-1])
    st.sidebar.markdown("---")

    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please add it to the model folder.", icon="⚠️")
    else:
        model = load_model(cfg_model_path)
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)

        if st.sidebar.checkbox("Custom Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        input_option = st.sidebar.radio("Select input type: ", ['Image', 'Video'])
        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload data from the local system'])

        if input_option == 'Image':
            image_input(data_src)
        else:
            video_input(data_src)

    # List of detected objects
    if hasattr(model, "results") and len(model.results.pred[0]):
        st.markdown("## Detected Objects")
        detected_objects = set(model.results.names[0])
        st.write(list(detected_objects))

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
