# Emotion-Aware Movie Recommender 🎬😊

**A movie recommendation system that reads your facial expression through your webcam and suggests films that match your mood.**

🎓 BSc thesis, Department of Computer Science & Telecommunications, University of Thessaly
🚧 **Status:** in progress. Emotion recognition is working; the recommendation engine is under development.

<!-- Add a GIF or screenshot of webcam_test.py running: drag it into this file while editing on GitHub -->

---

## 💡 How It Works

```
 Webcam frame ──► Face detection ──► CNN emotion classifier ──► Mood-to-movie mapping ──► Recommendations
                 (OpenCV Haar)       (7 emotions, Keras)        (in development)
```

1. **Capture:** OpenCV reads live frames from the webcam
2. **Detect:** a Haar Cascade classifier finds faces in each frame
3. **Classify:** each face is cropped, resized to 48×48 grayscale, and passed to a custom CNN that predicts one of 7 emotions: *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*
4. **Recommend:** the detected emotional state will be mapped to personalized movie suggestions *(in development)*

---

## 🧠 Model

A custom 4-block convolutional neural network built with TensorFlow/Keras and trained from scratch on the **FER2013** facial expression dataset.

| Component | Details |
|---|---|
| Input | 48×48 grayscale face images |
| Architecture | 4 convolutional blocks (64 → 128 → 256 → 512 filters), each with two Conv2D layers, Batch Normalization, ReLU, MaxPooling, and Dropout |
| Classifier head | Dense (512) + Batch Normalization + Dropout (0.4) → Softmax over 7 classes |
| Augmentation | Random horizontal flip, rotation, and zoom |
| Optimizer | Adam (learning rate 0.001) with `ReduceLROnPlateau` scheduling |
| Loss | Categorical cross-entropy |
| Training | 40 epochs, batch size 64 |

<!-- Add your results here, e.g.:
**Test accuracy:** XX%
Include a training curve or confusion matrix image if you have one.
-->

---

## 🛠️ Tech Stack

| Area | Technology |
|---|---|
| Deep learning | TensorFlow, Keras |
| Computer vision | OpenCV |
| Machine learning | scikit-learn |
| Language | Python |
| Dataset | FER2013 |

---

## 💻 Getting Started

### 1. Install

```bash
git clone https://github.com/GeorgeTsagg18/Movie-Recommender-Thesis.git
cd Movie-Recommender-Thesis
pip install -r requirements.txt
```

### 2. Try the live emotion detector

A trained model (`emotion_model.h5`) is included, so you can test it right away:

```bash
python webcam_test.py
```

A window opens showing your webcam feed with the detected emotion and confidence above your face. Press **`q`** to quit.

### 3. (Optional) Retrain the model

Download the [FER2013 dataset from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the project folder like this:

```
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── ...
│   └── surprise/
└── test/
    ├── angry/
    └── ...
```

Then run:

```bash
python train_emotion_model.py
```

The new model is saved as `emotion_model.h5`.

---

## 📁 Project Structure

```
Movie-Recommender-Thesis/
├── train_emotion_model.py   # Builds and trains the CNN on FER2013
├── webcam_test.py           # Real-time face detection and emotion inference
├── emotion_model.h5         # Trained model weights
└── requirements.txt
```

---

## 🗺️ Roadmap

- [x] Train a CNN for 7-class facial emotion recognition
- [x] Real-time webcam emotion detection
- [ ] Design the emotion-to-genre/movie mapping algorithm
- [ ] Integrate a movie database for recommendations
- [ ] Build a user interface for the full system

---

## 👤 Author

**George Tsagkarakis**
[GitHub](https://github.com/GeorgeTsagg18) · [Upwork](https://www.upwork.com/freelancers/~0193d0fe5d3fd14798) · [Email](mailto:georgetsag18@gmail.com)
