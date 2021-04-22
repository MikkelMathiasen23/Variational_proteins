import torch
import numpy as np 
from ray import tune
from ray.tune import track
from scipy.special import erfinv
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
import torch
import numpy as np
from misc import data, c
from torch import optim
from scipy.stats import spearmanr
from torch.distributions.normal import Normal
from HVAE import HVAE
import os 

os.chdir('/content/drive/MyDrive/Mathematical modelling and computation/2. semester/02460 - Advanced machine learning/variational-proteins-main/Variational_proteins')

def train_func(config):
  batch_size = 128
  os.chdir('/content/drive/MyDrive/Mathematical modelling and computation/2. semester/02460 - Advanced machine learning/variational-proteins-main/Variational_proteins')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataloader, df, mutants_tensor, mutants_df, weights, neff = data(batch_size = batch_size, device = device)
  wildtype   = dataloader.dataset[0] # one-hot-encoded wildtype 
  eval_batch = torch.cat([wildtype.unsqueeze(0), mutants_tensor])


  kwargs = {      
      'alphabet_size': dataloader.dataset[0].shape[0],
      'seq_len':       dataloader.dataset[0].shape[1],
      'device': device,
      'hidden_size': 2000,
      'shared_size': 4,
      'repeat': 1,
      'group_sparsity': False,
       'layers': config['config']['layers'],
      'latents': config['config']['latents'],
  }

  step = 0.1
  alpha_warm_up = torch.arange(0,1+step,step)
  vae   = HVAE(**kwargs).to(device)
  opt   = optim.Adam(vae.parameters())

  # rl  = Reconstruction loss
  # kl  = Kullback-Leibler divergence loss
  # cor = Spearman correlation to experimentally measured 
  #       protein fitness according to eq.1 from paper
  stats = { 'rl': [], 'kl': [], 'cor': [] }

  for epoch in range(200):
      # Unsupervised training on the MSA sequences.
      vae.train()
      if epoch > len(alpha_warm_up)-1:
        k = len(alpha_warm_up)-1
      else:
        k = epoch
      
      epoch_losses = { 'rl': [], 'kl': [] }
      for batch in dataloader:
          opt.zero_grad()
          x_hat = vae(batch)
          loss, rl, kl      = vae.loss(x_hat, batch,alpha_warm_up =alpha_warm_up[k])
          loss.mean().backward()
          opt.step()
          epoch_losses['rl'].append(rl.mean().item())
          epoch_losses['kl'].append(kl.mean().item())

      # Evaluation on mutants
      vae.eval()
      x_hat_eval = vae(eval_batch)
      elbos, _, _ = vae.loss(x_hat_eval, eval_batch, alpha_warm_up =alpha_warm_up[k])
      diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
      cor, _      = spearmanr(mutants_df.value, diffs.cpu().detach())
      # Populate statistics 
      stats['rl'].append(np.mean(epoch_losses['rl']))
      stats['kl'].append(np.mean(epoch_losses['kl']))
      stats['cor'].append(np.abs(cor))
      track.log(rho =np.abs(cor))


      to_print = [
          f"{c.HEADER}EPOCH %03d"          % epoch,
          f"{c.OKBLUE}RL=%4.4f"            % stats['rl'][-1], 
          f"{c.OKGREEN}KL=%4.4f"           % stats['kl'][-1], 
          f"{c.OKCYAN}|rho|=%4.4f{c.ENDC}" % stats['cor'][-1]
      ]
      print(" ".join(to_print))

from itertools import combinations_with_replacement
param_grid = []
for l in range(2,5):
  latent_combination = list(combinations_with_replacement([4,8,16], l))
  layer_combination = list(combinations_with_replacement([256,512], l))
  for layer in layer_combination:
    for ll in latent_combination: 
      param_grid.append({'layers':(layer),'latents':(ll)})

analysis = tune.run(
    train_func,
    config={ 
        "config": tune.grid_search(param_grid)
    },
    local_dir="/content/drive/MyDrive/Mathematical modelling and computation/2. semester/02460 - Advanced machine learning/variational-proteins-main/Variational_proteins",
    resources_per_trial={'gpu':1})

print("Best config: ", analysis.get_best_config(
    metric="rho", mode="max"))
