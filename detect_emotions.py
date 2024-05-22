import cv2
from utils.facial_detection import detect_faces
from utils.emotion_analysis import predict_emotion

def main():
    # Load pre-trained model
    emotion_model = load_emotion_model()

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Predict emotion for each face
        for face in faces:
            x, y, w, h = face
            roi = frame[y:y+h, x:x+w]
            emotion = predict_emotion(roi, emotion_model)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
