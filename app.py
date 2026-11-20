import streamlit as st
from PIL import Image
from src.predict import load, predict

st.title("Plant Disease Detection")

model, classes = load()
file = st.file_uploader("Завантажте фото листка", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img)
    label, prob = predict(img, model, classes)
    st.write(f"Результат: {label} ({prob:.1%})")
