import sys
sys.path.insert(1, './src')
import numpy as np
import joblib
from src.predict import predict_output
from src.train import train_model
from src.dataload import load_data
from src.rbm import RBM
from src.preprocessing import preprocess_data


train_datasets, test_datasets = load_data("ml-100k/u")
train_folds, test_folds = preprocess_data(
    train_datasets, test_datasets, verbose=True)


inp = str(input("Do you want to train the model again? ")).lower()
if inp == 'y' or inp == 'yes':
    rbm = train_model(train_folds, test_folds, nh=100, k_walk=20,
                      epochs=50, batch_size=32, lr=0.01, verbose=True)
else:
    rbm = joblib.load("rbm.pkl")

liked = np.array(
    list(set(map(int, input("Enter ID of movies you liked: ").split()))))
disliked = np.array(
    list(set(map(int, input("Enter ID of movies you hated: ").split()))))

predict_output(rbm, train_folds, liked=liked, disliked=disliked,
               top_like=3, top_dislike=5, verbose=True)