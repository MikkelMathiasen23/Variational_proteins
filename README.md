# Variational proteins

This project is based on replicating the results *"Deep generative models of genetic variation capture the effects of mutations"* (https://www.nature.com/articles/s41592-018-0138-4). A Variational Autoencoder (VAE) is trained unsupervised on sequence data and thereby learning the latent representation sequence families. The latent space can be used to investigate mutations in sequence families. 

## The data 
The data used for replicating the results can be found in the `data` folder containing three excel spreadsheets with sequence data. In the `misc.py` file these files are parsed and one-hot-encoded, and weights for sequence weighting are calculated. Sequence weighting are performed in order to compensate for overrepresentation in the data. 

## The model
The implementation of the VAE can be found in the `VAE.py` file. The VAE can be used in multiple settings, which can be selected in the `train.py` file. By selecting `group_sparsity = True` (default) the last layer in the decoder are sparse meaning that each hidden unit can only be affected by a few, thereby creating a sparse structure. The number of parameters to learn in the sparse structure can further be reduced by repeating the same elements of the matrix this can be selected by `repeat = 1` (default). 
If group sparsity is not performed the last layer in the decoder is a normal fully-connected layer. 

Furthermore, the VAE can be used in a Bayesian setting or normal setting. By setting `bayesian = True` (default) the layers in the decoder (and the group-sparsity parameters) are modelled as Gaussian distributions, thereby sampling weights (and bias) resulting in a regularized network. This results in an extra KL-divergence term based on the parameters as it is wished that the parameters are close to a prior distribution. 

## Training
The model can be trained using by: `from train import train_vae` in the Jupyter notebook: `notebook.ipynb`. The train function can also be found in `train.py`. The `train_vae` function contains different parameter settings for the VAE training: `epochs`, `batch_size`, .... 

After training a file depending whether the setting was bayesian or not:  `trained.model.bayesian.pth` or `trained.model.non_bayesian.pth` containing training statistics, model parameters which can be further explored in the `notebook.ipynb`. In the Jupyter notebook some initial plotting of the statistics can also be found. 
