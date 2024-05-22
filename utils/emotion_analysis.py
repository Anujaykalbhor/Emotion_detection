import numpy as np

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(roi, model):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))

    preds = model.predict(roi)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    return emotion
