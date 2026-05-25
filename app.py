import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.title("Object Detection")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)

    result_img = results[0].plot()

    st.image(result_img, caption="Detected Objects")
