import sys
sys.path.insert(1, './src')
import numpy as np
import joblib
from src.predict import predict_output
from src.train import train_model
from src.dataload import load_data
from src.rbm import RBM
from src.preprocessing import preprocess_data

print("Welcome to my Recommendation project. Check out my handles:-\n\n[linkedin.com/in/aitik-gupta][github.com/aitikgupta][kaggle.com/aitikgupta]\n\n")

train_datasets, test_datasets = load_data("ml-100k/u")
train_folds, test_folds = preprocess_data(
    train_datasets, test_datasets, verbose=True)


inp = str(input("\nDo you want to train the model again? ")).lower()
if inp == 'y' or inp == 'yes':
    rbm = train_model(train_folds, test_folds, nh=1000, k_walk=100,
                      epochs=10, batch_size=32, lr=0.01, verbose=True)
else:
    rbm = joblib.load("src/rbm.pkl")

liked = np.array(
    list(set(map(int, input("\nEnter ID of movies you liked: ").split()))))
disliked = np.array(
    list(set(map(int, input("Enter ID of movies you hated: ").split()))))

predict_output(rbm, train_folds, liked=liked, disliked=disliked,
               top_like=3, top_dislike=3, verbose=True)