import argparse
import os
import shutil

import numpy as np
import pandas as pd
import yaml

from utils.find_project_root import find_project_root


# Process Pokémon and one-hot encode the types
def preprocess_training_labels(dataset) -> pd.DataFrame:
    pokemon = pd.read_csv(PROJECT_ROOT / SOURCE_DIRECTORY / dataset)
    pokemon = pokemon[["pokedex_number", "name", "type1", "type2"]]

    # Create one-hot columns for each type
    df_one_hot = pd.get_dummies(pokemon.type1.str.capitalize(), prefix='is', prefix_sep='')+pd.get_dummies(pokemon.type2.str.capitalize(), prefix='is', prefix_sep='')
    df_one_hot = df_one_hot.clip(0, 1)
    pokemon = pd.merge(pokemon, df_one_hot, left_index=True, right_index=True)
            
    # Save output
    pokemon.to_csv(PROJECT_ROOT / DESTINATION_DIRECTORY / 'pokemon.csv', index=False)            
    return(pokemon)

# Process image data
def preprocess_training_data(dataset) -> pd.DataFrame:

    data_directory_images = PROJECT_ROOT / SOURCE_DIRECTORY / dataset
    output_directory = PROJECT_ROOT / DESTINATION_DIRECTORY / "pokemon"

    pokemon = pd.read_csv(PROJECT_ROOT / DESTINATION_DIRECTORY / 'pokemon.csv')
    pokemon["imagePath"] = np.nan

    # Remove processed folder and create empty new one
    try:
        shutil.rmtree(output_directory)
        os.mkdir(output_directory)
    except:
        os.mkdir(output_directory)

    # Copy images to processed folder
    for image in os.listdir(data_directory_images):
        pokemon_id = image.split('.')[0]

        # Add leading zeroes to ID
        pokemon_id = pokemon_id.zfill(3)

        # Images with no variety (e.g. "211.png")
        if pokemon_id.isnumeric():

            # Copy to processed folder
            src = data_directory_images / image
            dst = os.path.join(output_directory, pokemon_id + ".png")
            shutil.copyfile(src, dst)

    # Set image path in data frame
    pokemon['imagePath'] = pokemon['pokedex_number'].apply(lambda x: os.path.join(output_directory, str(x).zfill(3) + '.png'))

    # Drop Pokemon without image path
    pokemon = pokemon.dropna(subset=["imagePath"])
    
    # Save pokemon.csv with image paths
    pokemon.to_csv(PROJECT_ROOT / DESTINATION_DIRECTORY / 'pokemon-with-image-paths.csv', index=False)
    
    return(pokemon)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)

    PROJECT_ROOT = find_project_root()

    SOURCE_DIRECTORY: str = params['data_preprocess']['source_directory']
    DESTINATION_DIRECTORY: str = params['data_preprocess']['destination_directory']
    TRAIN_DATA_LABELS: str = params['data_preprocess']['dataset_labels']
    TRAIN_DATA_IMAGES: str = params['data_preprocess']['dataset_images']

    pokemon = preprocess_training_labels(TRAIN_DATA_LABELS)
    pokemon = preprocess_training_data(TRAIN_DATA_IMAGES)

    print(pokemon.head())