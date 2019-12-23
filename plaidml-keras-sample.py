#!/usr/bin/env python

import numpy as np
import os
import time


import keras
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Model, Sequential
import time

def main():
    # load cifar10
    x_train, y_train, x_test, y_test = load_dataset()
    model = generate_model()
    # print model
    model.summary()
    # train
    start = time.time()
    model.fit(x_train, y_train, batch_size=1025, epochs=10, validation_split=0.1)
    end = time.time()
    elapsed = end - start
    print("Training: elasped time is " + str(elapsed))

    pt = []
    # predict
    for i in range(5):
        start_ev = time.time()
        score = model.evaluate(x_test, y_test)
        end_ev = time.time()
        print("score: ", score)
        elapsed_ev = end_ev - start_ev
        print("Evaluate: elasped time is " + str(elapsed_ev))
        pt.append(elapsed_ev)
    predict_time = np.array(pt)
    print("predict_elapsed time: ", predict_time)
    print("average: ", np.average(predict_time))
    print("std: ", np.std(predict_time))


def generate_model():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

def load_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test /255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    main()

