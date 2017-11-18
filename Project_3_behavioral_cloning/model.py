import os
import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Convolution2D, Dense, Flatten, Dropout

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def augmentation(images, steering_angles):
    # flip image and measurement
    images_flipped = [cv2.flip(img, 1) for img in images]
    steering_angles_flipped = [-1 * s for s in steering_angles]

    images_augment = images + images_flipped
    steering_angles = steering_angles + steering_angles_flipped

    # change BGR to YUV
    images_augment = [cv2.cvtColor(img, cv2.COLOR_BGR2YUV) for img in images]
    return np.array(images_augment), np.array(steering_angles)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            X_train, y_train = augmentation(images, angles)
            yield shuffle(X_train, y_train)

def CnnModel(input_shape=(160, 320, 3)):
    model = Sequential()
    # ROI extraction
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=input_shape))
    # normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # conv1
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    # conv2
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    # conv3
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    # conv4
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # conv5
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())

    # FC1
    model.add(Dense(100))
    model.add(Dropout(.5))
    # FC2
    model.add(Dense(50))
    model.add(Dropout(.5))
    # FC3
    model.add(Dense(10))
    # Output
    model.add(Dense(1))
    return model

def main():

    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # model define and training
    model = CnnModel()
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
    
    # save model
    model.save('model.h5')
    print("model saved")

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()