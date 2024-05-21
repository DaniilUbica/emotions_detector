import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import layers, models

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

def guess_fer_emotion(img):
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    from fer import FER
    

    emo_detector = FER(mtcnn=False)
    dominant_emotion, _ = emo_detector.top_emotion(img)
    return dominant_emotion