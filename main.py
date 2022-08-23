import json

import mlflow
import logging
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
   "test_regression_model"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the mlflow experiment and AWS access
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    EXPERIMENT_NAME = "dl_model_demo"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info("pipeline experiment_id: %s", experiment.experiment_id)

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "download" in active_steps:
            download_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "get_data"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            file_path_uri = download_run.data.params['local_folder']
            logger.info('downloaded data is located locally in folder: %s', file_path_uri)
            logger.info(download_run)
        

        if "basic_cleaning" in active_steps:
    
            basic_cleaning_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "../../data/sample1.csv",
                    "output_artifact": "../../data/clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )
            basic_cleaning_run_id = basic_cleaning_run.run_id
            basic_cleaning_run = mlflow.tracking.MlflowClient().get_run(basic_cleaning_run_id)
            logger.info(basic_cleaning_run)
        

        if "data_check" in active_steps:
            data_check_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "../../data/clean_sample.csv",
                    "ref": "../../data/clean_sample.csv",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                }
            )
            data_check_run_run_id = data_check_run.run_id
            data_check_run = mlflow.tracking.MlflowClient().get_run(data_check_run_run_id)
            logger.info(data_check_run)


        if "data_split" in active_steps:
            data_split_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split"),
                "main",
                parameters={
                    "input": "../../data/clean_sample.csv",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )

            data_split_run_id = data_split_run.run_id
            data_split_run = mlflow.tracking.MlflowClient().get_run(data_split_run_id)
            logger.info(data_split_run)

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
            train_random_forest_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "../../data/trainval_data.csv",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"
                }
            )

            train_random_forest_run_id = train_random_forest_run.run_id
            train_random_forest_run = mlflow.tracking.MlflowClient().get_run(train_random_forest_run_id)
            logger.info(train_random_forest_run)

        if "test_regression_model" in active_steps:
            test_regression_model_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": "/Users/tuananh/tuananh/nd0821-c2-build-model-workflow-starter/src/train_random_forest/random_forest_dir",
                    "test_dataset": "/Users/tuananh/tuananh/nd0821-c2-build-model-workflow-starter/data/test_data.csv"
                }
            )

            test_regression_model_run_id = test_regression_model_run.run_id
            test_regression_model_run = mlflow.tracking.MlflowClient().get_run(test_regression_model_run_id)
            logger.info(test_regression_model_run)

    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)

if __name__ == "__main__":
    go()
