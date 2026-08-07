import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('best_model.h5')
edge_case_folder = 'edge_cases'

for fname in os.listdir(edge_case_folder):
    path = os.path.join(edge_case_folder, fname)
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = model.predict(arr, verbose=0)[0][0]
    label = "GENUINE" if prob > 0.5 else "FAKE"
    confidence = prob if prob > 0.5 else 1 - prob
    print(f"{fname}: {label} (confidence: {confidence*100:.1f}%)")