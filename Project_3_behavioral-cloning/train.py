import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Convolution2D, Dense, Flatten

def extract_data(logfile_path):
    '''
    extract image and measurements from driving log .csv file
    :param logfile_path: the path of log file
    :return: images BGR image arrays
             measurements steering angle vectors
    '''
    lines = []
    with open(logfile_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines[1:]:
        source_path = line[0]
        file_name = source_path.split('/')[-1]
        image_path = 'data/IMG/' + file_name
        image = cv2.imread(image_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    return images, measurements

def augmentation(images, steering_angles):
    # flip image and measurement
    images_flipped = [cv2.flip(img, 1) for img in images]
    steering_angles_flipped = [-1 * s for s in steering_angles]

    images = images + images_flipped
    steering_angles = steering_angles + steering_angles_flipped

    # change BGR to YUV
    images_augment = [cv2.cvtColor(img, cv2.COLOR_BGR2YUV) for img in images]
    return np.array(images_augment), np.array(steering_angles)

def CnnModel(input_shape):
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
    # FC2
    model.add(Dense(50))
    # FC3
    model.add(Dense(10))
    # Output
    model.add(Dense(1))
    return model

def main():

    # data preprocessing and augmentation
    images, steering_angles = extract_data('data/driving_log.csv')
    x_train, y_train = augmentation(images, steering_angles)

    # model define and training
    model = CnnModel(x_train[0].shape)
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1)

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    # save model
    model.save('model.h5')
    print("model saved")

if __name__ == '__main__':
    main()