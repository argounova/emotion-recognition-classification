import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Load the training data
emotion_ds = tf.keras.utils.image_dataset_from_directory(
  'data/emotions',
  image_size=(150, 150),
  batch_size=32
)

# Get the class names
class_names = emotion_ds.class_names
print(class_names)