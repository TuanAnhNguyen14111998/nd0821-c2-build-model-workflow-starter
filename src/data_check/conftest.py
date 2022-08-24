import pytest
import pandas as pd
import os


def pytest_addoption(parser):
    parser.addoption("--folder_path", action="store")
    parser.addoption("--artifact_name", action="store")
    parser.addoption("--storage_id", action="store")
    parser.addoption("--train_pos_num", action="store")
    parser.addoption("--train_neg_num", action="store")
    parser.addoption("--test_pos_num", action="store")
    parser.addoption("--test_neg_num", action="store")


@pytest.fixture(scope='session')
def data_train(request):
    storage_id = request.config.option.storage_id
    data_path = f"../../data/information/{storage_id}/dataset/train.csv" 

    if os.path.isfile(data_path) == False:
        pytest.fail(f"You must download artifact from s3://mlflow/{storage_id}")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def storage_id(request):
    storage_id = request.config.option.storage_id

    return storage_id


@pytest.fixture(scope='session')
def data_test(request):
    storage_id = request.config.option.storage_id
    data_path = f"../../data/information/{storage_id}/dataset/test.csv" 

    if os.path.isfile(data_path) == False:
        pytest.fail(f"You must download artifact from s3://mlflow/{storage_id}")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def train_pos_num(request):
    train_pos_num = request.config.option.train_pos_num

    if train_pos_num is None:
        pytest.fail("You must provide a train_pos_num")

    return int(train_pos_num)


@pytest.fixture(scope='session')
def train_neg_num(request):
    train_neg_num = request.config.option.train_neg_num

    if train_neg_num is None:
        pytest.fail("You must provide a train_neg_num")

    return int(train_neg_num)


@pytest.fixture(scope='session')
def test_pos_num(request):
    test_pos_num = request.config.option.test_pos_num

    if test_pos_num is None:
        pytest.fail("You must provide a test_pos_num")

    return int(test_pos_num)


@pytest.fixture(scope='session')
def test_neg_num(request):
    test_neg_num = request.config.option.test_neg_num

    if test_neg_num is None:
        pytest.fail("You must provide a test_neg_num")

    return int(test_neg_num)
