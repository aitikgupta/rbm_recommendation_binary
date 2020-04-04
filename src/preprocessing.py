import numpy as np
import pandas as pd
import torch
from dataload import load_data


def binarify(value):
    if value >= 3:
        return 1
    return 0


def convert(dataset, users, movies):
    dataset = dataset.loc[:, [0, 1, 2]]
    dataset.loc[:, 2] = dataset.loc[:, 2].apply(lambda x: binarify(x))
    data = []
    for user in range(users):
        id_movies = dataset.loc[:, 1][dataset[0] == user+1].values
        id_ratings = dataset.loc[:, 2][dataset[0] == user+1].values
        ratings = np.array([-1 for movie in range(movies)])
        ratings[id_movies - 1] = id_ratings
        data.append(ratings)
    data = np.array(data)
    return data


def preprocess(train_data, test_data):
    max_userid_train = train_data.loc[:, 0].max()
    max_movieid_train = train_data.loc[:, 1].max()
    max_userid_test = test_data.loc[:, 0].max()
    max_movieid_test = test_data.loc[:, 1].max()

    n_users = max(max_userid_train, max_userid_test)
    n_movies = max(max_movieid_train, max_movieid_test)

    train_processed = convert(train_data, n_users, n_movies)
    test_processed = convert(test_data, n_users, n_movies)
    train_processed = torch.FloatTensor(train_processed)
    test_processed = torch.FloatTensor(test_processed)
    return train_processed, test_processed


def preprocess_data(train_datasets, test_datasets, verbose=True):
    train_folds = []
    test_folds = []
    for fold, (train_dataset, test_dataset) in enumerate(zip(train_datasets, test_datasets)):
        train_processed, test_processed = preprocess(
            train_dataset, test_dataset)
        train_folds.append(train_processed)
        test_folds.append(test_processed)
    if verbose:
        print("[INFO] Preprocessing Complete!")
    return train_folds, test_folds


if __name__ == "__main__":
    train_datasets, test_datasets = load_data("../ml-100k/u")
    train_folds, test_folds = preprocess_data(
        train_datasets, test_datasets, verbose=True)
    for fold in range(5):
        print(
            f"Preprocessed shape of {fold} fold: {np.array(train_folds[fold]).shape}")