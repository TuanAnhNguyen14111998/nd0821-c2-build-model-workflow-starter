#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import mlflow
import logging
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with mlflow.start_run(run_name="train_val_test_split") as mlrun:
        logger.info(f"Fetching artifact {args.input}")

        # Download input artifact. This will also note that this script is using this
        # particular version of the artifact
        logger.info(f"Fetching artifact {args.input}")
        artifact_local_path = args.input

        df = pd.read_csv(artifact_local_path)

        logger.info("Splitting trainval and test")
        trainval, test = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
        )

        # Save to output files
        for df, k in zip([trainval, test], ['trainval', 'test']):
            logger.info(f"Uploading {k}_data.csv dataset")
            df.to_csv(f"../../data/{k}_data.csv", index=False)

        mlflow.log_artifacts("../../data/", artifact_path="data")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
