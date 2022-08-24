import json

import mlflow
import logging
import os
import hydra
from omegaconf import DictConfig

_steps = [
    "get_data",
    # "basic_cleaning",
    "data_check",
    # "data_split",
    "train_model",
    # "test_regression_model"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    os.environ['MLFLOW_TRACKING_URI'] = 'postgresql+psycopg2://db_user:123@localhost/mlflow_db'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

    EXPERIMENT_NAME = "tracking_experiments"
    try:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f's3://{config["etl"]["bucket_name"]}')
        mlflow.set_experiment(EXPERIMENT_NAME)
    except:
        mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info("pipeline experiment_id: %s", experiment.experiment_id)

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "get_data" in active_steps:
            download_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "get_data"),
                "main",
                parameters={
                    "folder_path": config["etl"]["folder_path"],
                    "path_info": config["etl"]["path_info"],
                    "artifact_name": config["etl"]["artifact_name"],
                    "artifact_type": config["etl"]["artifact_type"],
                    "storage_id": config["etl"]["storage_id"],
                    "artifact_description": "Upload or Download Dataset"
                }
            )
            client = mlflow.tracking.MlflowClient()
            download_run = client.get_run(download_run.run_id)
            logger.info(download_run)
        
        if "basic_cleaning" in active_steps:
    
            # basic_cleaning_run = mlflow.run(
            #     os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
            #     "main",
            #     parameters={
            #         "input_artifact": "../../data/sample1.csv",
            #         "output_artifact": "../../data/clean_sample.csv",
            #         "output_type": "clean_sample",
            #         "output_description": "Data with outliers and null values removed",
            #         "min_price": config['etl']['min_price'],
            #         "max_price": config['etl']['max_price']
            #     },
            # )
            # basic_cleaning_run_id = basic_cleaning_run.run_id
            # basic_cleaning_run = mlflow.tracking.MlflowClient().get_run(basic_cleaning_run_id)
            # logger.info(basic_cleaning_run)

            pass
        

        if "data_check" in active_steps:
            data_check_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "folder_path": config["etl"]["folder_path"],
                    "artifact_name": config["etl"]["artifact_name"],
                    "storage_id": config["etl"]["storage_id"],
                    "train_pos_num": config["data_check"]["train_pos_num"],
                    "train_neg_num": config['data_check']['train_neg_num'],
                    "test_pos_num": config["data_check"]["test_pos_num"],
                    "test_neg_num": config['data_check']['test_neg_num'],
                }
            )
            data_check_run_run_id = data_check_run.run_id
            data_check_run = mlflow.tracking.MlflowClient().get_run(data_check_run_run_id)
            logger.info(data_check_run)


        if "data_split" in active_steps:
            # data_split_run = mlflow.run(
            #     os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split"),
            #     "main",
            #     parameters={
            #         "input": "../../data/clean_sample.csv",
            #         "test_size": config["modeling"]["test_size"],
            #         "random_seed": config["modeling"]["random_seed"],
            #         "stratify_by": config["modeling"]["stratify_by"]
            #     }
            # )

            # data_split_run_id = data_split_run.run_id
            # data_split_run = mlflow.tracking.MlflowClient().get_run(data_split_run_id)
            # logger.info(data_split_run)
            pass

        if "train_model" in active_steps:

            model_config = os.path.abspath("model_config.json")
            with open(model_config, "w+") as fp:
                json.dump(dict(config["modeling"]["densenet121"].items()), fp)

            train_model_run = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_model"),
                "main",
                parameters={
                    "folder_path": config["etl"]["folder_path"],
                    "artifact_name": config["etl"]["artifact_name"],
                    "storage_id": config["etl"]["storage_id"],
                    "image_size": config["modeling"]["image_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "model_config": model_config,
                    "output_artifact": config["modeling"]["output_artifact"]
                }
            )

            train_model_run_id = train_model_run.run_id
            train_model_run = mlflow.tracking.MlflowClient().get_run(train_model_run_id)
            logger.info(train_model_run)

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
