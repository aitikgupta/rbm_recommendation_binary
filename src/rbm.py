import torch
import numpy as np
from preprocessing import preprocess_data
from dataload import load_data
from tqdm import tqdm

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        ph_given_v = torch.sigmoid(activation)
        return ph_given_v, torch.bernoulli(ph_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        pv_given_h = torch.sigmoid(activation)
        return pv_given_h, torch.bernoulli(pv_given_h)

    def update(self, v0, vk, ph0, phk, lr):
        self.W += lr*((torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t())
        self.b += lr*(torch.sum((v0 - vk), 0))
        self.a += lr*(torch.sum((ph0 - phk), 0))

    def kwalk(self, k_walk, v0):
        vk = v0
        for k in range(k_walk):
            _, hk = self.sample_h(vk)
            _, vk = self.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = self.sample_h(vk)
        return vk, phk

    def loss(self, v0, vk):
        return float(torch.mean(torch.abs(v0[v0 >= 0]-vk[v0 >= 0])))

    def fit(self, X, batch_size=8, epochs=5, k_walk=10, lr=0.5, verbose=True):
        val_losses = []
        for epoch in tqdm(range(1, epochs+1), position=0, leave=True):
            train_loss = 0.
            counter = 0
            for dp in range(0, len(X)-batch_size, batch_size):
                v = X[dp:dp+batch_size]
                if len(X)-dp < batch_size:
                    v = X[dp:]
                ph0, _ = self.sample_h(v)
                vk, phk = self.kwalk(k_walk=k_walk, v0=v)
                train_loss += self.loss(v, vk)
                self.update(v, vk, ph0, phk, lr=lr)
                counter += 1
            loss = train_loss/counter
            val_losses.append(loss)
            if verbose:
                tqdm.write(f"Epoch: {epoch}\tLoss: {str(loss)}")
        return val_losses

    def test(self, X, y, verbose=True):
        test_loss = 0.
        counter = 0
        for dp in range(0, len(y)):
            v = X[dp:dp+1]
            vt = y[dp:dp+1]
            if len(vt[vt >= 0]) > 0:
                _, h = self.sample_h(v)
                _, vk = self.sample_v(h)
                test_loss += self.loss(vt, vk)
                counter += 1
        loss = test_loss/counter
        if verbose:
            print(f"Test Loss: {str(loss)}")
        return loss

    def predict(self, X, y):
        ph, h = self.sample_h(y)
        pv, v = self.sample_v(h)
        return pv, v


if __name__ == "__main__":
    train_datasets, test_datasets = load_data("../ml-100k/u")
    train_folds, test_folds = preprocess_data(
        train_datasets, test_datasets, verbose=True)
    nv = train_folds[0].shape[1]
    nh = [100, 500, 1000]
    epochs = [10, 50, 100]
    k_walks = [1, 5, 20, 100]
    lr = [0.0001, 0.001, 0.01, 1]
    losses = []
    for e in epochs:
        for k in k_walks:
            for l in lr:
                for h in nh:
                    rbm = RBM(nv=nv, nh=h)
                    for f in range(len(train_folds)):
                        losses = rbm.fit(
                            train_folds[f], epochs=e, batch_size=32, lr=l, k_walk=k, verbose=False)
                        test_loss = rbm.test(
                            train_folds[f], test_folds[f], verbose=False)
                        print(
                            f"{test_loss} at Epochs: {e}, K_walks:{k}, LR: {l}, N_Hidden_Nodes: {h}, Fold: {f}")
                        losses.append(test_loss)
                    del rbm
    print(f"Min Loss: {min(losses)}")