import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('best_model.h5')

# Test with a genuine image from your dataset
genuine_folder = 'split_dataset/test/genuine'
fake_folder = 'split_dataset/test/fake'

genuine_img_name = os.listdir(genuine_folder)[0]
fake_img_name = os.listdir(fake_folder)[0]

def test_image(path, expected):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    label = "GENUINE" if pred > 0.5 else "FAKE"
    print(f"Expected: {expected} | Got: {label} | Raw score: {pred:.4f}")

print("Testing genuine image:")
test_image(os.path.join(genuine_folder, genuine_img_name), "GENUINE")

print("Testing fake image:")
test_image(os.path.join(fake_folder, fake_img_name), "FAKE")