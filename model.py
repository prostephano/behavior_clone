# External
import csv
import psutil
import sys
import numpy
import sklearn
import cv2
import matplotlib
import matplotlib.pyplot as plt
import keras.models

from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import load_model

# Internal
from augmentation import *
from balance import balance_data

BATCH_SIZE = 32
RESIZED_SHAPE = (80, 160)

def get_all_data(csv_loc = './driving_log.csv'):
    to_return = []

    with open(csv_loc) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            to_return.append(line)

    return to_return

def show_image(image):
    plt.imshow(image)
    plt.show()

def pretrain_image_process(image_path):
    image = cv2.imread(image_path)
    return image

def center_generator(samples, batch_size, perform_aug):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        numpy.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            img_index = 0

            # center img loc, left img loc, right img loc, steer angle, throttle, break, speed
            for batch_sample in batch_samples:
                image = pretrain_image_process(batch_sample[img_index])
                angle = float(batch_sample[3])

                if (image is None):
                    continue

                if (perform_aug == False) :
                    # original image
                    images.append(image)
                    angles.append(angle)
                else:
                    image, angle = behavior_perform_aug(image, angle)
                    images.append(image)
                    angles.append(angle)

            X_train = numpy.array(images)
            y_train = numpy.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def mean_normalize(x):
    return K.mean(x/127.5 - 1., axis=3, keepdims=True)

def train_get_initializer():
    return keras.initializers.RandomNormal()

def train_get_regularizer():
    return keras.regularizers.l2(0.)

def train_network(train_set, validation_set, image_shape, batch_size):
    train_set_generator = center_generator(train_set, batch_size, True)
    validation_set_generator = center_generator(validation_set, batch_size, False)

    model = keras.models.Sequential()

    # Start of Preprocess
    # Grayscale AND normalize
    model.add(Lambda(mean_normalize, input_shape=image_shape, output_shape=(image_shape[0], image_shape[1], 1)))

    # Crop
    crop_dimension=((65,15), (0,0))
    model.add( Cropping2D(cropping=crop_dimension) )

    #end of pre process

    # Conv section
    model.add(Convolution2D(filters=6, kernel_size=5, strides=1, padding='valid', data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=6, kernel_size=5, strides=1, padding='valid', data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=6, kernel_size=5, strides=1, padding='valid', data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # end of Conv section

    # Flatten
    model.add(keras.layers.core.Flatten())
    # End of flatten

    # Start of Dense section
    model.add(keras.layers.core.Dense(160, kernel_initializer=train_get_initializer(), \
            kernel_regularizer=train_get_regularizer()))
    model.add(keras.layers.Activation('relu'))
    model.add(Dropout(0.5))

    model.add(keras.layers.core.Dense(80, kernel_initializer=train_get_initializer(), \
            kernel_regularizer=train_get_regularizer()))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.core.Dense(30, kernel_initializer=train_get_initializer(), \
            kernel_regularizer=train_get_regularizer()))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.core.Dense(10, kernel_initializer=train_get_initializer(), \
            kernel_regularizer=train_get_regularizer()))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.core.Dense(1))
    model.add(keras.layers.Activation('linear'))
    # End of Dense section

    # Loss
    model.compile(loss='mse', optimizer='adam')

    # call backs
    filepath="model_{epoch:02d}-{val_loss:.2f}.h"
    callbacks = [ \
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min', period=1), \
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1, mode='min'), \
        ]

    # Training
    model.fit_generator(train_set_generator, steps_per_epoch=int(len(train_set)), \
            validation_data=validation_set_generator, \
            nb_val_samples=len(validation_set), nb_epoch=1000, callbacks=callbacks)

    # Conclude
    model.save('model.h5')

def main():
    samples = get_all_data()
    samples = balance_data(samples)
    numpy.random.shuffle(samples)

    train_samples, validation_samples = train_test_split(samples, test_size=0.05)
    validation_samples = copy.copy(train_samples)

    image_shape = numpy.shape(pretrain_image_process(train_samples[0][0]))
    train_network(train_samples, validation_samples, image_shape, BATCH_SIZE)

if __name__ == '__main__':
    main()
