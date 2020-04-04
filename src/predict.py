import torch
import numpy as np
import joblib
from dataload import load_data
from preprocessing import preprocess_data


def predict_output(model, train_folds, liked, disliked, top_like=3, top_dislike=2, verbose=True):
    user_data = np.array([-1 for movie in range(train_folds[0].shape[1])])
    user_data[liked-1] = 1
    user_data[disliked-1] = 0
    user_data = user_data.reshape(1, -1)
    probs = np.zeros(train_folds[0].shape[1])
    outs = np.zeros(train_folds[0].shape[1])
    for fold in range(len(train_folds)):
        probabilities, outputs = model.predict(train_folds[fold], torch.FloatTensor(user_data))
        probabilities = np.array(probabilities).ravel()
        outputs = np.array(outputs).ravel()
        probs += probabilities
        outs += outputs
    probs /= float(len(train_folds))
    outs /= float(len(train_folds))
    outs = (outs > 0.5)
    for i, v in enumerate(outs):
        if v:
            outs[i] = 1
        else:
            outs[i] = 0
    like = []
    dislike = []
    _ = [like.append((i+1,probs[i])) for i in range(len(outs)) if outs[i] == 1]
    _ = [dislike.append((i+1,probs[i])) for i in range(len(outs)) if outs[i] == 0]
    like.sort(key = lambda x: x[1], reverse=1)
    dislike.sort(key = lambda x: x[1], reverse=1)
    if verbose:
        print("\nYou'll LIKE these Movies with ID:")
        _ = [print(f"MovieID: {i[0]}, Confidence: {i[1]}") for i in like[: min(top_like, len(like))]]
        print("\nYou'll DISLIKE these Movies with ID:")
        _ = [print(f"MovieID: {i[0]}, Confidence: {i[1]}") for i in dislike[:min(top_dislike, len(dislike))]]
    return like[:min(top_like, len(like))], dislike[:min(top_dislike, len(dislike))]


if __name__ == "__main__":
    train_datasets, test_datasets = load_data("../ml-100k/u")
    train_folds, test_folds = preprocess_data(train_datasets, test_datasets, verbose=True)
    model = joblib.load("rbm.pkl")
    liked = np.array([5,8,10,29,35,53])
    disliked = np.array([1,7,50,32,100])
    like, dislike = predict_output(model, train_folds, liked, disliked, top_like=5, top_dislike=3)