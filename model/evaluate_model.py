import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = tf.keras.models.load_model('best_model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'split_dataset/test', target_size=(224, 224), batch_size=32,
    class_mode='binary', shuffle=False)

loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc*100:.2f}%")

pred_probs = model.predict(test_gen)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = test_gen.classes

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=list(test_gen.class_indices.keys())))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))