import argparse
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.find_project_root import find_project_root


# Load training images
def load_training_data(labels) -> np.array:
    train_image = []

    for i in tqdm(range(labels.shape[0])):

        img = tf.keras.utils.load_img(labels.iloc[i]["imagePath"], color_mode='rgba')
        img = tf.keras.utils.img_to_array(img)
        img = img/255
        train_image.append(img)
    X = np.array(train_image)
    
    return(X)
    
# Create labels
def create_labels(labels):
    return(pokemon[["is" + POKEMON_TYPE_TRAIN]])

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)

    PROJECT_ROOT = find_project_root()

    SEED: str = params['base']['seed']
    POKEMON_TYPE_TRAIN: str = params['base']['pokemon_type_train']

    DESTINATION_DIRECTORY: str = params['data_preprocess']['destination_directory']
    MODEL_TEST_SIZE: float = params['train']['test_size']

    pokemon = pd.read_csv(PROJECT_ROOT / DESTINATION_DIRECTORY / "pokemon-with-image-paths.csv")

    X = load_training_data(pokemon)
    y = create_labels(pokemon)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, test_size=MODEL_TEST_SIZE, stratify=y)

    pickle.dump(X, open(PROJECT_ROOT / DESTINATION_DIRECTORY / "X.pckl", "wb"))
    pickle.dump(X_train, open(PROJECT_ROOT / DESTINATION_DIRECTORY / "X_train.pckl", "wb"))
    pickle.dump(X_test, open(PROJECT_ROOT / DESTINATION_DIRECTORY / "X_test.pckl", "wb"))

    pickle.dump(y, open(PROJECT_ROOT / DESTINATION_DIRECTORY / "y.pckl", "wb"))
    pickle.dump(y_train, open(PROJECT_ROOT / DESTINATION_DIRECTORY / "y_train.pckl", "wb"))
    pickle.dump(y_test, open(PROJECT_ROOT / DESTINATION_DIRECTORY / "y_test.pckl", "wb"))