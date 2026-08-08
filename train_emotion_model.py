import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# 1. Φόρτωση δεδομένων από τους φακέλους

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    'fer2013/train',
    color_mode='grayscale',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'  # Δημιουργει encoding για τα 7 συναισθηματα
)

print("Loading test data")
test_ds = tf.keras.utils.image_dataset_from_directory(
    'fer2013/test',
    color_mode='grayscale',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# 2. Βελτιστοποίηση και κανονικοποίηση (Normalization)
# Διαιρούμε τις τιμές των pixels με το 255 για να παμε στο [0,1]
normalization_layer = layers.Rescaling(1./255)

# Εφαρμόζουμε κανονικοποίηση στα datasets
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# 3. Χτίσιμο του CNN
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),
    # 1o στρώμα φίλτρων
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # 2o στρώμα φίλτρων
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # 3o στρώμα φίλτρων
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Τελικό στάδιο απόφασης
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax') # 7 συναισθήματα
])

# 4. Μεταγλώττιση και εκπαίδευση
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Beginning training...")
epochs = 30

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs
)

# Αποθήκευση του μοντέλου
model.save('emotion_model.h5')
print("The model has been succesfully saved as 'emotion_model.h5'!")