import cv2
import numpy as np

from model import get_emotions_list, guess_fer_emotion

def show_video():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    # model = tf.keras.models.load_model('models/model.keras')

    i = 0
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            i += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
            emotion = guess_fer_emotion(frame)
            cv2.putText(frame, emotion, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def guess_emotion(model, img):
    image = img
    image = cv2.resize(image, (48,48))
    image = np.invert(np.array([image]))
    output = model.predict(image)
    emotions = get_emotions_list()
    return emotions[np.argmax(output)][0]