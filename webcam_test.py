import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Φόρτωση του εκπαιδευμένου μοντέλου
model = load_model('emotion_model.h5')

# Έλεγχος των καναλιών που περιμένει το μοντέλο (1 ή 3)
input_channels = model.input_shape[-1]
print(f"The model has been loaded! Awaiting input with {input_channels} color channels.")

# 2. Αλφαβητική σειρά συναισθημάτων (όπως ταξινομούνται από την TensorFlow)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 3. Αλγόριθμος Haar Cascade για εντοπισμό προσώπου
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. Έναρξη της κάμερας
cap = cv2.VideoCapture(0)
print("The camera has been activated! Press 'q' inside the window to end the test.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't connect.")
        break

    # Μετατροπή σε ασπρόμαυρη εικόνα για τον εντοπισμό προσώπου
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Σχεδιασμός πλαισίου γύρω από το πρόσωπο
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 1. Αποκοπή της περιοχής του προσώπου
        roi_color = frame[y:y+h, x:x+w]

        # 2. Μετατροπή BGR -> RGB (όπως εκπαιδεύτηκε η TensorFlow)
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

        # 3. Προσαρμογή μεγέθους σε 48x48 pixels
        roi_resized = cv2.resize(roi_rgb, (48, 48), interpolation=cv2.INTER_AREA)

        # 4. Μετατροπή σε float32
        roi = roi_resized.astype('float32')

        # 5. Προσαρμογή καναλιών αν το μοντέλο εκπαιδεύτηκε σε Grayscale
        if input_channels == 1:
            roi = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
            roi = np.expand_dims(roi, axis=-1)

        roi = np.expand_dims(roi, axis=0)  # Μορφή (1, 48, 48, C)

        # Πρόβλεψη συναισθήματος
        prediction = model.predict(roi, verbose=0)
        max_index = np.argmax(prediction)
        label = emotion_labels[max_index]
        confidence = prediction[0][max_index] * 100

        # Εκτύπωση αποτελεσμάτων στο terminal για testing
        print(f"Πρόβλεψη: {label} ({confidence:.1f}%) | Raw: {np.round(prediction[0], 2)}")

        # Εμφάνιση συναισθήματος πάνω από το πλαίσιο
        text = f"{label} ({confidence:.0f}%)"
        cv2.putText(frame, text, (int(x), int(max(0, y - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Εμφάνιση του παραθύρου βίντεο
    cv2.imshow('Live Emotion Detector', frame)

    # Τερματισμός με το κουμπί q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Απελευθέρωση πόρων
cap.release()
cv2.destroyAllWindows()