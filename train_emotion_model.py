import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Κρύβει τα warnings του TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

# Επιτάχυνση ροής δεδομένων στη μνήμη
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# 2. Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), # καθέφτρισμα εικόνας
    layers.RandomRotation(0.1), # περιστροφή εικόνας έως 10% δεξιά ή αριστερά
    layers.RandomZoom(0.1), # zoom in / zoom out έως 10%
])

# 3. Custom 4-Block CNN
model = keras.Sequential([
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
    layers.Dropout(0.2),

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
    layers.Dropout(0.3),

    # Στάδιο απόφασης
    layers.Flatten(),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),
    layers.Dense(7, activation='softmax')
])

# 4. Μεταγλώττιση και Callbacks
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), # όρισα ρυθμό μάθησης για να ανεβεί το accuracy
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback για να μειώνεται δυναμικά το Learning Rate
lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

# 5. Εκπαίδευση
epochs = 40
print("Begining training on the 4-Block CNN...")
history = model.fit(
    train_ds,
    validation_data = test_ds,
    epochs = epochs,
    callbacks = [lr_reduction]
)

# 6. Αποθήκευση μοντέλου
model.save('emotion_model.h5')
print("\nThe Model has been saved successfully as 'emotion_model.h5'!")