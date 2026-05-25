import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Object Detection and Tracking")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    results = model(img_array)

    plotted = results[0].plot()

    st.image(plotted, caption="Detected Objects")
