import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load model
model = load_model("mnist_model.h5")

st.title("✍️ Draw a Digit (MNIST Recognizer)")

st.write("Draw a digit below and click Predict")

# ---------------- CANVAS ---------------- #
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------------- PREPROCESS ---------------- #
def preprocess(canvas_data):
    img = canvas_data.astype("uint8")

    # Convert to grayscale (already black/white but safe)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28 (MNIST format)
    resized = cv2.resize(gray, (28, 28))

    # Normalize
    resized = resized / 255.0

    # Reshape for model
    resized = resized.reshape(1, 28, 28)

    return resized

# ---------------- PREDICT ---------------- #
if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = preprocess(canvas_result.image_data)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediction: {digit}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        st.bar_chart(prediction[0])