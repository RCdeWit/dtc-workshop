import argparse
import pickle

import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from tensorflow import keras

from utils.find_project_root import find_project_root

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)

    PROJECT_ROOT = find_project_root()
    DESTINATION_DIRECTORY: str = params['data_preprocess']['destination_directory']

    # Load model
    # estimator = pickle.loads((PROJECT_ROOT / "outputs" / "model.pckl").read_bytes())
    model = keras.models.load_model(PROJECT_ROOT / "outputs" / "model")

    # Load data
    X = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "X.pckl").read_bytes())
    X_train = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "X_train.pckl").read_bytes())
    X_test = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "X_test.pckl").read_bytes())

    y = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "y.pckl").read_bytes())
    y_train = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "y_train.pckl").read_bytes())
    y_test = pickle.loads((PROJECT_ROOT / DESTINATION_DIRECTORY / "y_test.pckl").read_bytes())

    # Predict all Pokémon
    predictions = model.predict(X) > 0.5

    # Calculate metrics
    metrics = {}

    metrics["acc"] = float(accuracy_score(y, predictions))
    metrics["precision"] = float(precision_score(y, predictions))
    metrics["recall"] = float(recall_score(y, predictions))
    metrics["f1"] = float(f1_score(y, predictions))

    # Save metrics
    with open(PROJECT_ROOT / "outputs" / "metrics.yaml", 'w') as file:
        yaml.dump(metrics, file, default_flow_style=False)

    # Plot confusion matrix
    cm = confusion_matrix(y, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save confusion matrix
    plt.savefig(PROJECT_ROOT / "outputs" / "confusion_matrix.png", dpi=150, bbox_inches='tight', pad_inches=0)

    print(f"Evaluation done!")
    print(metrics)