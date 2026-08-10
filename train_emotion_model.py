import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
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

# 2. Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), # καθέφτρισμα εικόνας
    layers.RandomRotation(0.1), # περιστροφή εικόνας έως 10% δεξιά ή αριστερά
    layers.RandomZoom(0.1), # zoom in / zoom out έως 10%
])

# 3. Χτίσιμο του custom CNN (4 conv blocks)
model = keras.Sequential([
    # Είσοδος
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Rescaling(1./255),
    data_augmentation,

    # Block 1 (64 φίλτρα)
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2 (128 φίλτρα)
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3 (256 φίλτρα)
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 4 (512 φίλτρα)
    layers.Conv2D(512, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(512, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Στάδιο απόφασης
    layers.Flatten(),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),

    # Έξοδος για τα 7 συναισθήματα
    layers.Dense(7, activation='softmax')
])

# 4. Μεταγλώττιση και callbacks
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), # όρισα ρυθμός μάθησης για να ανεβεί το accuracy
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks για προσαρμογή της ταχύτητας μάθησης αν σταματήσει να βελτιώνεται η ακρίβεια
lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

# 5. Εκπαίδευση

epochs = 40 # Τις αύξησα εφόσον το δίκτυο είναι πιο βαθύ πλέον

print("Beginning training...")
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    callbacks=[lr_reduction]
)

# Αποθήκευση του μοντέλου
model.save('emotion_model.h5')
print("The model has been succesfully saved as 'emotion_model.h5'!")