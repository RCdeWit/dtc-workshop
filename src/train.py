import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras import layers, regularizers
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras

from utils.find_project_root import find_project_root


def compile_model(model_image_size_x, model_image_size_y):
    img_input = layers.Input(shape=(model_image_size_x, model_image_size_y, 4))

    model = Sequential()

    model.add(Conv2D(4, kernel_size=(5,5), activation='relu', kernel_regularizer=regularizers.l2(l=0.01), input_shape=(model_image_size_x, model_image_size_y, 4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(4, kernel_size=(5,5), activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(8, activation="relu"))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=MODEL_LEARNING_RATE) #Adam, RMSprop or SGD

    model.compile(
        loss='binary_crossentropy'
        , optimizer=optimizer
        , metrics=[keras.metrics.AUC()]
    #     , metrics=[keras.metrics.Recall()]
    )

    model.summary()

    return(model)

def train_estimator(model):
    def calculate_class_weights(y_train):
        ratio_true = sum(y_train["is" + POKEMON_TYPE_TRAIN] == 1) / len(y_train["is" + POKEMON_TYPE_TRAIN])
        ratio_false = sum(y_train["is" + POKEMON_TYPE_TRAIN] != 1) / len(y_train["is" + POKEMON_TYPE_TRAIN])

        return {0: ratio_true, 1: ratio_false}


    estimator = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        class_weight= calculate_class_weights(y_train),
                        epochs=MODEL_EPOCHS, 
                        batch_size=MODEL_BATCH_SIZE,
                        verbose=1)

    return(estimator)

def save_estimator(estimator):
    # Training history
    plt.figure()
    plt.ylabel('Loss / Accuracy')
    plt.xlabel('Epoch')

    for k in estimator.history.keys():
        plt.plot(estimator.history[k], label = k) 
    plt.legend(loc='best')

    plt.savefig(PROJECT_ROOT / "outputs" / "train_history.png", dpi=150, bbox_inches='tight', pad_inches=0)

    # Save model itself
    model.save(PROJECT_ROOT / "outputs" / "model")

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)

    PROJECT_ROOT = find_project_root()
    DESTINATION_DIRECTORY: str = params['data_preprocess']['destination_directory']
    POKEMON_TYPE_TRAIN: str = params['base']['pokemon_type_train']

    MODEL_LEARNING_RATE: float = params['train']['learning_rate']
    MODEL_EPOCHS: int = params['train']['epochs']
    MODEL_BATCH_SIZE: int = params['train']['batch_size']

    X = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "X.pckl").read_bytes())
    X_train = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "X_train.pckl").read_bytes())
    X_test = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "X_test.pckl").read_bytes())

    y = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "y.pckl").read_bytes())
    y_train = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "y_train.pckl").read_bytes())
    y_test = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "y_test.pckl").read_bytes())

    model_image_size_x = len(X[1])
    model_image_size_y = len(X[2])

    model = compile_model(model_image_size_x, model_image_size_y)

    estimator = train_estimator(model)
    save_estimator(estimator)