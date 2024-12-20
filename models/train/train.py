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
import warnings
warnings.filterwarnings("ignore")



# Load the dataset
# emotion_ds = 'data/emotions'
sports_train_ds = 'data/sports/train'
sports_test_ds = 'data/sports/test'
sports_valid_ds = 'data/sports/valid'


# Load the training data
train_ds = tf.keras.utils.image_dataset_from_directory(
  sports_train_ds,
  labels='inferred',
  label_mode='categorical',
  class_names=None,
  color_mode='rgb',
  batch_size=20,
  image_size=(224, 224),
  shuffle=True,
  seed=None,
  validation_split=None,
  subset=None,
  interpolation='bilinear',
  follow_links=False, 
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  sports_valid_ds,
  labels='inferred',
  label_mode='categorical',
  class_names=None,
  color_mode='rgb',
  batch_size=20,
  image_size=(224, 224),
  shuffle=True,
  seed=None,
  validation_split=None,
  subset=None,
  interpolation='bilinear',
  follow_links=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
  sports_test_ds,
  labels='inferred',
  label_mode='categorical',
  class_names=None,
  color_mode='rgb',
  batch_size=20,
  image_size=(224, 224),
  shuffle=True,
  seed=None,
  validation_split=None,
  subset=None,
  interpolation='bilinear',
  follow_links=False,
)

#  Get the class names and display the images
plt.rcParams['figure.figsize']= (20,8)
class_names = train_ds.class_names
plt.figure(figsize=(20,12))
for x in train_ds.take(1):
    for i in range(15):
        plt.subplot(3,5,i+1)
        image = x[0][i] / 255.
        plt.imshow(image)
        label_index = np.argmax(x[1][i].numpy())
        plt.title(class_names[label_index])
        plt.axis('off')
plt.show()

# Optimize the datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = Sequential([
  layers.RandomFlip('horizontal', input_shape=(224, 224, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.1),
])

# Create the model
model = keras.Sequential([
  data_augmentation,
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(128, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(256, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(512, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(100, activation='softmax')
])

# Build the model
# model = Sequential([
#   data_augmentation,
#   layers.Rescaling(1./255),

#   layers.Conv2D(64, (3,3), activation='relu'),
#   layers.BatchNormalization(),
#   layers.MaxPooling2D((2,2)),
#   layers.Dropout(0.2),

#   layers.Conv2D(128, (3,3), activation='relu'),
#   layers.BatchNormalization(),
#   layers.MaxPooling2D((2,2)),
#   layers.Dropout(0.2),
  
#   layers.Conv2D(128, (3,3), activation='relu'),
#   layers.BatchNormalization(),
#   layers.MaxPooling2D((2,2)),
#   layers.Dropout(0.4),

#   layers.GlobalAveragePooling2D(),
#   layers.Flatten(),
#   layers.Dense(512, activation='relu'),
#   layers.BatchNormalization(),
#   layers.Dropout(0.5),

#   layers.Dense(256, activation='relu'),
#   layers.BatchNormalization(),
#   layers.Dropout(0.5),

#   layers.Dense(128, activation='relu'),
#   layers.BatchNormalization(),
#   layers.Dropout(0.5),

#   layers.Dense(100, activation='softmax')
# ])

# Compile the model
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

print(model.summary())

# Train the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10,
  verbose=1,
  batch_size=20
)

# Visualize the training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

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
model.save('sports_model.h5')

# Generate a classification report