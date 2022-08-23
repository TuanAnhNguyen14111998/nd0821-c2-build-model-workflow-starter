#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with mlflow.start_run(run_name="test_model") as mlrun:
        logger.info("Downloading artifacts")
        # Download input artifact. This will also log that this script is using this
        # particular version of the artifact
        model_local_path = args.mlflow_model

        # Download test dataset
        test_dataset_path = args.test_dataset

        # Read test dataset
        X_test = pd.read_csv(test_dataset_path)
        y_test = X_test.pop("price")

        logger.info("Loading model and performing inference on test set")
        
        sk_pipe = mlflow.sklearn.load_model(model_local_path)
        y_pred = sk_pipe.predict(X_test)

        logger.info("Scoring")
        r_squared = sk_pipe.score(X_test, y_test)

        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"Score: {r_squared}")
        logger.info(f"MAE: {mae}")

        mlflow.log_metric(key="r2", value=r_squared)
        mlflow.log_metric(key="mae", value=mae)
        mlflow.log_param("mlflow run id", mlrun.info.run_id)
        mlflow.set_tag('pipeline_step', __file__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str, 
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
