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
    types = set(pokemon["type1"])
    for t in types:
        pokemon["is" + str(t).capitalize()] = 0

    # Iterate over Pokémon
    for i, p in pokemon.iterrows():

        #  Set one-hot columns to 1 for relevant types
        pokemon.loc[i, "is" + p["type1"].capitalize()] = 1

        if not pd.isna(p["type2"]):
            pokemon.loc[i, "is" + p["type2"].capitalize()] = 1
            
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
        while len(pokemon_id) < 3:
            pokemon_id = "0" + pokemon_id

        # Images with no variety (e.g. "211.png")
        if pokemon_id.isnumeric():

            # Copy to processed folder
            src = data_directory_images / image
            dst = os.path.join(output_directory, pokemon_id + ".png")
            shutil.copyfile(src, dst)

            # Set image path in data frame
            pokemon.loc[pokemon["pokedex_number"] == int(pokemon_id), 'imagePath'] = dst

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