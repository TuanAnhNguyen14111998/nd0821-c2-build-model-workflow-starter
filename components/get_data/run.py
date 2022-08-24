#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import mlflow
import os

from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    path_info = args.path_info.split(",")
    with mlflow.start_run(run_name="get_data") as mlrun:
        if args.storage_id == "":
            mlflow.log_param("artifact_name", args.artifact_name)
            mlflow.log_param("artifact_type", args.artifact_type)
            mlflow.log_param("folder_storage", f"s3://mlflow/{mlrun.info.run_id}")
            mlflow.log_param("mlflow_run_id", mlrun.info.run_id)
            mlflow.log_param("description", "Upload data raw")
            mlflow.set_tag('pipeline_step', __file__)
            mlflow.log_artifact(path_info[0], artifact_path="dataset")
            mlflow.log_artifact(path_info[1], artifact_path="dataset")

            logger.info("finished upload data to %s", f"s3://mlflow/{mlrun.info.run_id}")
        else:
            folder_path = f"../../data/information/{args.storage_id}/"
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            client = mlflow.tracking.MlflowClient()
            client.download_artifacts(
                run_id=args.storage_id,
                path="dataset",
                dst_path=folder_path
            )
            mlflow.log_param("artifact_name", args.artifact_name)
            mlflow.log_param("artifact_type", args.artifact_type)
            mlflow.log_param("folder_storage", f"s3://mlflow/{args.storage_id}")
            mlflow.log_param("mlflow_run_id", mlrun.info.run_id)
            mlflow.log_param("description", f"Download data from s3://mlflow/{args.storage_id}")
            mlflow.set_tag('pipeline_step', __file__)

            logger.info("finished download data to %s", folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("folder_path", type=str, help="Folder store images")

    parser.add_argument("path_info", type=str, help="List/File csv for training & testing model")

    parser.add_argument("artifact_name", type=str, help="List artifact name for tracking with S3")

    parser.add_argument("artifact_type", type=str, help="Type of list artifact")

    parser.add_argument("storage_id", type=str, help="")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)
