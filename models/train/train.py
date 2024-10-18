import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
# from keras import to_categorical
# from sklearn.metrics import classification_report, confusion_matrix



# Load the dataset
emotion_ds = 'data/emotions'


# Load the training data
train_ds = tf.keras.utils.image_dataset_from_directory(
  emotion_ds,
  labels='inferred',
  label_mode='int',
  color_mode='rgb',
  batch_size=32,
  image_size=(144, 144),
  shuffle=True,
  seed=123,
  validation_split=0.2,
  subset='training',
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  emotion_ds,
  labels='inferred',
  label_mode='int',
  color_mode='rgb',
  batch_size=32,
  image_size=(144, 144),
  shuffle=True,
  seed=123,
  validation_split=0.2,
  subset='validation',
)

# class_labels = train_ds.class_names
# plt.figure(figsize=(15,10))
# shown_classes = set()
# for images, labels in train_ds.take(1):
#   for i in range(len(images)):
#     class_name = class_labels[labels[i]]
#     if class_name in shown_classes:
#       ax = plt.subplot(1, 6, len(shown_classes) + 1)
#       plt.imshow(images[i].numpy().astype("uint8"))
#       plt.title(class_name) 
#       plt.axis("off")
#       shown_classes.add(class_name)
#     if len(shown_classes) == len(class_labels):
#       break
  
# plt.tight_layout()
# plt.show()



# Get the class names
# class_names = emotion_ds.class_names

# Split the dataset into training, validation, and test datasets
# def get_partitions(
#   ds,
#   train_size=0.8,
#   val_size=0.1,
#   test_size=0.1,
#   shuffle=True,
#   shuffle_size=10000,
# ):
#   ds_size=len(ds)
#   if shuffle:
#     ds = ds.shuffle(shuffle_size, seed=12)
#   train_size = int(ds_size * train_size)
#   val_size = int(ds_size * val_size)
#   test_size = int(ds_size * test_size)
#   return ds.take(train_size), ds.skip(train_size).take(val_size), ds.skip(train_size + val_size).take(test_size)
  
# Get the training, validation, and test datasets
# train_ds, val_ds, test_ds = get_partitions(emotion_ds)

# Optimize the datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = Sequential([
  layers.RandomFlip('horizontal', input_shape=(144, 144, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.1),
])

# Build the model
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),

  layers.Conv2D(32, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  # layers.Dropout(0.2),

  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  
  # layers.Dropout(0.2),
  
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  # layers.Dropout(0.4),

  layers.GlobalAveragePooling2D(),
  layers.BatchNormalization(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

print(model.summary())

# Train the model
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  verbose=1,
  batch_size=32
)

# Visualize the training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save the model

# Generate a classification report