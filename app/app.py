import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ---- Page config ----
st.set_page_config(page_title="Currency Authenticator", layout="centered")
st.title("💵 Real-Time Currency Authentication")
st.write("Genuine vs Fake currency note detection using CNN")

# ---- Load TFLite model (cached so it loads only once) ----
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="currency_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)

def preprocess_image(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0   # same preprocessing as MobileNetV2 training
    img = np.expand_dims(img, axis=0)
    return img

def predict(frame):
    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prob = output[0][0]
    label = "GENUINE ✅" if prob > 0.5 else "FAKE ⚠️"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence

# ---- Sidebar controls ----
st.sidebar.header("Controls")
run_camera = st.sidebar.checkbox("Start Camera", value=False)
FRAME_WINDOW = st.image([])
result_placeholder = st.empty()
fps_placeholder = st.sidebar.empty()

# ---- Camera loop ----
if run_camera:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible. Check webcam connection.")
            break

        # Predict
        label, confidence = predict(frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
        result_placeholder.markdown(f"### Result: {label}  \nConfidence: {confidence*100:.1f}%")
        fps_placeholder.text(f"FPS: {fps:.1f}")

        # Streamlit re-run check (stop if checkbox unticked)
        run_camera = st.session_state.get("Start Camera", run_camera)

    cap.release()
else:
    st.info("Tick 'Start Camera' in the sidebar to begin real-time detection.")