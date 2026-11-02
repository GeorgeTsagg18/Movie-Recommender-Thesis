import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

from movie_recommender import MovieRecommender

# Αρχικοποίηση Flask App
app = Flask(__name__)

# 1. Φόρτωση Μοντέλου Συναισθημάτων
print("Φόρτωση του εκπαιδευμένου μοντέλου συναισθημάτων...")
emotion_model = load_model('emotion_model.h5')
input_channels = emotion_model.input_shape[-1]
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print(f"Το μοντέλο φορτώθηκε ({input_channels} channels).")

# 2. Φόρτωση Haar Cascade για ανίχνευση προσώπου
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Αρχικοποίηση Movie Recommender
recommender = MovieRecommender(data_dir='ml-latest-small', default_min_votes=20)

# Emojis και περιγραφές για κάθε συναίσθημα
EMOTION_META = {
    'Happy': {'emoji': '😄', 'label_el': 'Χαρά', 'color': '#22c55e'},
    'Sad': {'emoji': '😢', 'label_el': 'Λύπη', 'color': '#3b82f6'},
    'Angry': {'emoji': '😠', 'label_el': 'Θυμός', 'color': '#ef4444'},
    'Surprise': {'emoji': '😲', 'label_el': 'Έκπληξη', 'color': '#eab308'},
    'Fear': {'emoji': '😨', 'label_el': 'Φόβος', 'color': '#a855f7'},
    'Disgust': {'emoji': '🤢', 'label_el': 'Αηδία', 'color': '#f97316'},
    'Neutral': {'emoji': '😐', 'label_el': 'Ουδέτερο', 'color': '#94a3b8'}
}

def decode_base64_image(image_data):
    """Αποκωδικοποίηση εικόνας Base64 από τον browser σε OpenCV format (BGR)."""
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

@app.route('/')
def index():
    """Κεντρική σελίδα της εφαρμογής."""
    return render_template('index.html', emotion_meta=EMOTION_META)

@app.route('/api/analyze-and-recommend', methods=['POST'])
def analyze_and_recommend():
    """
    Endpoint που δέχεται το snapshot της κάμερας, εντοπίζει το πρόσωπο,
    προβλέπει το συναίσθημα και επιστρέφει απευθείας προτάσεις ταινιών.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Δεν στάλθηκε εικόνα'}), 400

        strategy = data.get('strategy', 'booster')
        if strategy not in ['booster', 'mirror']:
            strategy = 'booster'

        # Αποκωδικοποίηση εικόνας
        frame = decode_base64_image(data['image'])
        if frame is None:
            return jsonify({'success': False, 'error': 'Αποτυχία ανάγνωσης εικόνας'}), 400

        # Μετατροπή σε grayscale για εντοπισμό προσώπου
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({
                'success': False,
                'no_face': True,
                'error': 'Δεν εντοπίστηκε πρόσωπο. Κοίταξε καθαρά την κάμερα με καλό φωτισμό!'
            })

        # Επιλογή του μεγαλύτερου προσώπου στο κάδρο (αν υπάρχουν περισσότερα)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]

        # Προετοιμασία εισόδου για το CNN (48x48)
        if input_channels == 1:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
            roi_input = roi_resized.astype('float32')
            roi_input = np.expand_dims(roi_input, axis=-1)
        else:
            roi = frame[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (48, 48), interpolation=cv2.INTER_AREA)
            roi_input = roi_resized.astype('float32')

        roi_input = np.expand_dims(roi_input, axis=0) # (1, 48, 48, C)

        # Πρόβλεψη συναισθήματος
        preds = emotion_model.predict(roi_input, verbose=0)[0]
        max_idx = int(np.argmax(preds))
        dominant_emotion = emotion_labels[max_idx]
        confidence = float(preds[max_idx] * 100)

        # Όλα τα συναισθήματα με τα ποσοστά τους
        all_emotions = []
        for i, emo in enumerate(emotion_labels):
            all_emotions.append({
                'emotion': emo,
                'label_el': EMOTION_META[emo]['label_el'],
                'emoji': EMOTION_META[emo]['emoji'],
                'color': EMOTION_META[emo]['color'],
                'percentage': round(float(preds[i] * 100), 1)
            })
        # Ταξινόμηση κατά ποσοστό φθίνουσα
        all_emotions = sorted(all_emotions, key=lambda x: x['percentage'], reverse=True)

        # Λήψη προτάσεων ταινιών βάσει του κυρίαρχου συναισθήματος
        movies = recommender.recommend(
            emotion=dominant_emotion,
            strategy=strategy,
            top_n=9,
            diverse=True
        )

        return jsonify({
            'success': True,
            'emotion': {
                'name': dominant_emotion,
                'label_el': EMOTION_META[dominant_emotion]['label_el'],
                'emoji': EMOTION_META[dominant_emotion]['emoji'],
                'color': EMOTION_META[dominant_emotion]['color'],
                'confidence': round(confidence, 1),
                'breakdown': all_emotions
            },
            'strategy': strategy,
            'movies': movies
        })

    except Exception as e:
        print(f"Σφάλμα κατά την επεξεργασία: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recommend-manual', methods=['POST'])
def recommend_manual():
    """Endpoint για χειροκίνητη επιλογή συναισθήματος χωρίς κάμερα."""
    try:
        data = request.get_json()
        emotion = data.get('emotion', 'Happy')
        strategy = data.get('strategy', 'booster')
        movies = recommender.recommend(emotion=emotion, strategy=strategy, top_n=9, diverse=True)
        return jsonify({
            'success': True,
            'emotion': {
                'name': emotion,
                'label_el': EMOTION_META.get(emotion, {}).get('label_el', emotion),
                'emoji': EMOTION_META.get(emotion, {}).get('emoji', '🎬'),
                'color': EMOTION_META.get(emotion, {}).get('color', '#3b82f6')
            },
            'strategy': strategy,
            'movies': movies
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Τρέχει σε όλες τις διεπαφές (0.0.0.0) στη θύρα 5000 ώστε να ανοίγει απρόσκοπτα από τα Windows
    print("\n" + "="*60)
    print("🚀 Το Web App εκκινεί! Άνοιξε στον browser σου:")
    print("👉 http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

