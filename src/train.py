from preprocessing import preprocess_data
import matplotlib.pyplot as plt
from dataload import load_data
from rbm import RBM
import joblib


def train_model(train_folds, test_folds, nh=1000, k_walk=100, epochs=10, batch_size=32, lr=0.1, verbose=True):
    nv = train_folds[0].shape[1]
    rbm = RBM(nv=nv, nh=nh)
    if verbose:
        print(
            f"Hyperparameters:-\nn_hidden_nodes: {nh}, k_walks: {k_walk}, epochs: {epochs}, batch_size: {batch_size}, learning_rate: {lr}")
    for fold in range(len(train_folds)):
        print(f"Fold {fold}:-")
        losses = rbm.fit(train_folds[fold], epochs=epochs,
                         batch_size=batch_size, lr=lr, k_walk=k_walk, verbose=verbose)
        test_loss = rbm.test(train_folds[fold], test_folds[fold])
        plt.plot(losses, label=f"Fold {fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    if verbose:
        print("[INFO] Training Complete!")
        opt = str(input("[OPTION] Replace original rbm.pkl? ")).lower()
        if opt == "y" or opt == "yes":
            joblib.dump(rbm, "rbm.pkl")
            print("[INFO] Model Saved at src/rbm.pkl!")
        print("[INFO] Training Complete!")
    plt.title(f"Loss vs Epochs")
    plt.legend()
    plt.show()
    return rbm


if __name__ == "__main__":
    train_datasets, test_datasets = load_data("../ml-100k/u")
    train_folds, test_folds = preprocess_data(train_datasets, test_datasets)
    rbm = train_model(train_folds, test_folds)