import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from collections import deque
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Φόρτωση του εκπαιδευμένου μοντέλου
model = load_model('emotion_model.h5')

# Έλεγχος των καναλιών που περιμένει το μοντέλο (1 για Grayscale, 3 για RGB)
input_channels = model.input_shape[-1]
print(f"Το μοντέλο φορτώθηκε επιτυχώς! Αναμένει είσοδο με {input_channels} κανάλι/α χρώματος.")

# 2. Αλφαβητική σειρά συναισθημάτων (όπως ταξινομούνται από την TensorFlow)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Χρώματα για κάθε συναίσθημα (BGR format για OpenCV)
emotion_colors = {
    'Angry': (0, 0, 255),       # Κόκκινο
    'Disgust': (0, 140, 255),   # Πορτοκαλί
    'Fear': (128, 0, 128),      # Μωβ
    'Happy': (0, 255, 0),       # Πράσινο
    'Neutral': (200, 200, 200), # Ανοιχτό γκρι
    'Sad': (255, 0, 0),         # Μπλε
    'Surprise': (0, 255, 255)   # Κίτρινο
}

# 3. Αλγόριθμος Haar Cascade για εντοπισμό προσώπου
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buffer για εξομάλυνση (temporal smoothing) στα τελευταία 12 frames
prediction_buffer = deque(maxlen=12)

# 4. Έναρξη της κάμερας
cap = cv2.VideoCapture(0)
print("Η κάμερα ενεργοποιήθηκε! Πάτα 'q' στο παράθυρο για έξοδο.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Δεν ήταν δυνατή η σύνδεση με την κάμερα.")
        break

    # Μετατροπή σε ασπρόμαυρο για τον εντοπισμό προσώπου
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        # Αν δεν ανιχνεύεται πρόσωπο, καθαρίζουμε το ιστορικό
        prediction_buffer.clear()

    for (x, y, w, h) in faces:
        # Προετοιμασία εισόδου για το μοντέλο (48x48)
        if input_channels == 1:
            # Απευθείας αποκοπή από το grayscale frame (αποφυγή περιττών μετατροπών)
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
            roi_input = roi_resized.astype('float32')
            roi_input = np.expand_dims(roi_input, axis=-1)  # (48, 48, 1)
        else:
            roi_color = frame[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (48, 48), interpolation=cv2.INTER_AREA)
            roi_input = roi_resized.astype('float32')

        roi_input = np.expand_dims(roi_input, axis=0)  # (1, 48, 48, C)

        # Πρόβλεψη συναισθήματος
        raw_pred = model.predict(roi_input, verbose=0)[0]
        prediction_buffer.append(raw_pred)

        # Εξομάλυνση (Moving Average) για σταθερό αποτέλεσμα χωρίς τρεμόπαιγμα
        smoothed_pred = np.mean(prediction_buffer, axis=0)
        max_index = np.argmax(smoothed_pred)
        label = emotion_labels[max_index]
        confidence = smoothed_pred[max_index] * 100
        box_color = emotion_colors.get(label, (0, 255, 0))

        # Σχεδιασμός πλαισίου γύρω από το πρόσωπο με το χρώμα του συναισθήματος
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        # Πλαίσιο φόντου για το κείμενο ώστε να διαβάζεται καθαρά
        label_text = f"{label}: {confidence:.0f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_y = max(y - 10, text_h + 10)
        cv2.rectangle(frame, (x, text_y - text_h - 6), (x + text_w + 10, text_y + 4), box_color, -1)
        cv2.putText(frame, label_text, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Εμφάνιση πιθανοτήτων όλων των συναισθημάτων (Dashboard στην επάνω αριστερή γωνία)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (220, 190), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, (emo, prob) in enumerate(zip(emotion_labels, smoothed_pred)):
            prob_percent = prob * 100
            emo_text = f"{emo[:4]}: {prob_percent:4.1f}%"
            row_y = 35 + i * 22
            # Μπάρα πιθανότητας
            cv2.putText(frame, emo_text, (20, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            bar_len = int(prob * 90)
            bar_color = emotion_colors.get(emo, (255, 255, 255))
            cv2.rectangle(frame, (120, row_y - 10), (120 + bar_len, row_y - 2), bar_color, -1)

    # Εμφάνιση του παραθύρου βίντεο
    cv2.imshow('Live Emotion Detector (Press Q to quit)', frame)

    # Τερματισμός με το πλήκτρο 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Απελευθέρωση πόρων
cap.release()
cv2.destroyAllWindows()