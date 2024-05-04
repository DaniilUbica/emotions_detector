import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import layers, models
import cv2

train_data_path = 'faces/train/'
test_data_path = 'faces/test/'

def print_dataset_info():
    total = 0
    for expression in os.listdir(train_data_path):
        amount = len(os.listdir(train_data_path + expression))
        print(expression, ' ', amount)
        total += amount
    print('total = ', total)

def get_emotions_list():
    emotions = []
    for expression in os.listdir(train_data_path):
        data = [expression]
        emotions.append(data)
    return emotions

def show_images():
    i = 0
    plt.figure(figsize=(8, 8))
    for expression in os.listdir(train_data_path):
        image = load_img((train_data_path + expression) + '/' + os.listdir(train_data_path + expression)[0])
        plt.subplot(1, 7, i + 1)
        plt.imshow(image)
        plt.title(expression)
        plt.axis('off')
        i += 1
    plt.show()

def train_model():
    train_data_gen = ImageDataGenerator()
    train_dataset = train_data_gen.flow_from_directory(
            train_data_path, shuffle=True, target_size=(48, 48), color_mode='grayscale',
            class_mode='categorical',
            batch_size=128
        )
    
    test_data_gen = ImageDataGenerator()
    test_dataset = test_data_gen.flow_from_directory(
            test_data_path, shuffle=False, target_size=(48, 48), color_mode='grayscale',
            class_mode='categorical',
            batch_size=128
        )
    
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), padding = 'same', activation='relu', input_shape=(48,48,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (5,5), padding = 'same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, (3,3), padding = 'same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(512, (3,3), padding = 'same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(7, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, validation_data=test_dataset, epochs=50, batch_size=128, verbose=1)
    model.save('model.keras')

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
            # face_img = frame[y:y+h, x:x+w, 0]
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

## if my model works badly, use that
def guess_fer_emotion(img):
    from fer import FER
    
    emo_detector = FER(mtcnn=False)
    dominant_emotion, _ = emo_detector.top_emotion(img)
    return dominant_emotion

def main():
    show_video()

if __name__ == "__main__":
    main()