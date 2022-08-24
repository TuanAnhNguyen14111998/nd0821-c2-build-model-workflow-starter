from tokenize import String
from markupsafe import string
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def test_column_names(data_train: pd.DataFrame, data_test: pd.DataFrame):

    expected_colums = [
        "Images",
        "Classes"
    ]

    these_columns = data_train.columns.values

    assert list(expected_colums) == list(these_columns)

    these_columns = data_test.columns.values

    assert list(expected_colums) == list(these_columns)


def test_label_names(data_train: pd.DataFrame, data_test: pd.DataFrame):

    known_names = [0, 1]

    class_unique = set(data_train['Classes'].unique())

    assert set(known_names) == set(class_unique)

    class_unique = set(data_test['Classes'].unique())

    assert set(known_names) == set(class_unique)


def test_num_pos_neg(
    data_train: pd.DataFrame, data_test: pd.DataFrame,
    train_pos_num: int, train_neg_num: int,
    test_pos_num: int, test_neg_num: int):
    pos_num = data_train[data_train.Classes == 1].shape[0]
    assert pos_num == train_pos_num

    neg_num = data_train[data_train.Classes == 0].shape[0]
    assert neg_num == train_neg_num

    pos_num = data_test[data_test.Classes == 1].shape[0]
    assert pos_num == test_pos_num

    neg_num = data_test[data_test.Classes == 0].shape[0]
    assert neg_num == test_neg_num


def visualize_img(imgage_list, folder, option="train"):
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2
    for i in range(1, columns*rows +1):
        image = cv2.imread(f"../../data/images/{imgage_list[i-1]}")
        fig.add_subplot(rows, columns, i)
        plt.imshow(image)

    plt.savefig(f"../../data/information/{folder}/dataset/{option}.png")


def test_visualize_img(data_train: pd.DataFrame, data_test: pd.DataFrame, storage_id: string):
    
    image_train_list = data_train[:6].Images

    image_test_list = data_test[:6].Images

    visualize_img(image_train_list, storage_id, "train")
    visualize_img(image_test_list, storage_id, "test")

    assert os.path.isfile(f"../../data/information/{storage_id}/dataset/train.png") == True
    assert os.path.isfile(f"../../data/information/{storage_id}/dataset/test.png") == True
