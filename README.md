# Variational proteins

This project is based on replicating the results *"Deep generative models of genetic variation capture the effects of mutations"* (https://www.nature.com/articles/s41592-018-0138-4). A Variational Autoencoder (VAE) is trained unsupervised on sequence data and thereby learning the latent representation sequence families. The latent space can be used to investigate mutations in sequence families. A Ladder VAE models' latent representation of sequence families is also tested.

## The data 
The data used for replicating the results can be found in the `data` folder containing three excel spreadsheets with sequence data. In the `misc.py` file these files are parsed and one-hot-encoded, and weights for sequence weighting are calculated. Sequence weighting are performed in order to compensate for overrepresentation in the data. 

## The model :bug:
The implementations of the Vanilla VAE, Bayesian VAE and Ladder VAE can be found in the `models.py` file. The Bayesian VAE can be used in multiple settings, which can be selected in the `train_models.py` file. By selecting `group_sparsity = True` (default) the last layer in the decoder are sparse meaning that each hidden unit can only be affected by a few, thereby creating a sparse structure. The number of parameters to learn in the sparse structure can further be reduced by repeating the same elements of the matrix this can be selected by `repeat = 1` (default). If group sparsity is not performed the last layer in the decoder is a normal fully-connected layer. Furthermore, the VAE can be used in a Bayesian setting or normal setting. By setting `bayesian = True` (default) the layers in the decoder (and the group-sparsity parameters) are modelled as Gaussian distributions, thereby sampling weights (and bias) resulting in a regularized network. This results in an extra KL-divergence term based on the parameters as it is wished that the parameters are close to a prior distribution. 

Likewise for the Ladder VAE, hyperparameters and model architecture can be specified in `train_models.py`.

## Training :train:
The model can be trained using: `train_models.py`. In this script, hyperparameters should be selected and the model type to train should be specified. After the training is complete, a model state dict will be saved in the working directory containing training statistics and the given model. 

