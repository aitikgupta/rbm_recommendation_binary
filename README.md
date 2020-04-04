# Binary Recommendation System built from scratch using restricted Boltzmann machine (RBM)
#### Tech Stack:
*   NumPy
*   PyTorch

## Steps to reproduce:

Note: To maintain the ennvironments, I highly recommend using [conda](https://anaconda.org/).

```
git clone https://github.com/aitikgupta/rbm_recommendation_binary.git
cd rbm_recommendation_binary
conda env create -f environment.yml
conda activate {environment name, for eg. conda activate kaggle}
python main.py
```
Note: The project is user-driven, one can choose to train own model instead of self-trained model. One can choose the hyperparameters while training, the number of likes and dislikes to be predicted, even the verbosity.
#### [Input the IDs of movies you previously liked, and movies which you didn't like, and wait for it!]
---
### Each file in ```src``` folder can be run to test each component individually.
---
## About the model:
Research followed: [Restricted Boltzmann Machine](https://www.researchgate.net/publication/243463621_An_Introduction_to_Restricted_Boltzmann_Machines)
<br>However, I made some changes to the sampling and k-walks, which improved the prediction loss.
## Further Development:
*   Adding documentation to the functions, the RBM class
*   Make the verbosity uniform througout the codebase
*   Upgrade the binary recommendations to a scale of 1-5
### Thank You!
