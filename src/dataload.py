import pandas as pd

def load_data(path=None, verbose=True):
    if path == None:
        raise("Select a path to load data.")
    train_datasets = []
    test_datasets = []
    for fold in range(5):
        train_datasets.append(pd.read_csv(path+f"{fold+1}.base", delimiter="\t", header=None))
        test_datasets.append(pd.read_csv(path+f"{fold+1}.test", delimiter="\t", header=None))
    if verbose:
        print("[INFO] Dataset loaded Succesfully!")
    return train_datasets, test_datasets

if __name__ == "__main__":
    train, test = load_data("../ml-100k/u")