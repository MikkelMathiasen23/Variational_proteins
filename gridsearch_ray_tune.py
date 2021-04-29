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
from HVAE_v2 import HVAE
import os 

os.chdir('/zhome/45/c/127804/02460/Variational_proteins')

def train_func(config):
  batch_size = 128
  os.chdir('/zhome/45/c/127804/02460/Variational_proteins')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

  step = 0.05
  alpha_warm_up = torch.arange(0,1.5+step,step)
  vae   = HVAE(**args).to(device)
  opt   = optim.Adam(vae.parameters(), lr = 0.0005)

  # rl  = Reconstruction loss
  # kl  = Kullback-Leibler divergence loss
  # cor = Spearman correlation to experimentally measured 
  #       protein fitness according to eq.1 from paper
  stats = { 'rl': [], 'kl': [], 'cor': [] }
  
#################################### Initialize dicts for weights ############################################# 
  stats_encoder = {}
  stats_W = {}
  stats_reconstruct = {}
  for l in vae.e.keys():
    stats_encoder[l] = []
  for l in vae.W.keys():
    stats_W[l] = []
  cc = 0
  for i,l in enumerate(vae.reconstruct):
    if type(l) == torch.nn.Linear:
      name = 'reconstruct' + str(cc)
      stats_reconstruct[name]  = []
      cc+=1
#################################### Initialize dicts for mean and log ######################################### 
  stats_p_mu = {}
  stats_p_logvar = {}
  stats_e_mu = {}
  stats_e_logvar = {}
  stats_d_mu = {}
  stats_d_logvar = {}
  for i in range(len(args['layers'])):
        stats_d_mu['d'+str(i)] = []
        stats_d_logvar['d'+str(i)] = []
        stats_e_mu['e'+str(i)] = []
        stats_e_logvar['e'+str(i)] = []
        stats_p_mu['p'+str(i)] = []
        stats_p_logvar['p'+str(i)] = []
        

  for epoch in range(args['epochs']):
      # Unsupervised training on the MSA sequences.
      vae.train()
      if epoch > len(alpha_warm_up)-1:
        k = len(alpha_warm_up)-1
      else:
        k = epoch
      
      epoch_losses = { 'rl': [], 'kl': [] }
      for batch in dataloader:
          opt.zero_grad()
          x_hat,_, _,p_mu, p_logvar, d_mu, d_logvar = vae(batch)
          loss, rl, kl      = vae.loss(x_hat, batch,p_mu,p_logvar,d_mu,d_logvar,alpha_warm_up =1)
          loss.mean().backward()
          opt.step()
          epoch_losses['rl'].append(rl.mean().item())
          epoch_losses['kl'].append(kl.mean().item())

      # Evaluation on mutants
      vae.eval()
      x_hat_eval, e_mu, e_logvar,p_mu, p_logvar, d_mu, d_logvar = vae(eval_batch)
      elbos, _, _ = vae.loss(x_hat_eval, eval_batch,p_mu,p_logvar,d_mu,d_logvar, alpha_warm_up =1)#
      diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
      cor, _      = spearmanr(mutants_df.value, diffs.cpu().detach())
      
      # Populate statistics 
      stats['rl'].append(np.mean(epoch_losses['rl']))
      stats['kl'].append(np.mean(epoch_losses['kl']))
      stats['cor'].append(np.abs(cor))
      to_print = [
          f"{c.HEADER}EPOCH %03d"          % epoch,
          f"{c.OKBLUE}RL=%4.4f"            % stats['rl'][-1], 
          f"{c.OKGREEN}KL=%4.4f"           % stats['kl'][-1], 
          f"{c.OKCYAN}|rho|=%4.4f{c.ENDC}" % stats['cor'][-1]
      ]
      print(" ".join(to_print))
      for l in vae.e.keys():
        stats_encoder[l].append(torch.linalg.norm(vae.e[l].weight.data.cpu(), ord = 2).numpy())
      for l in vae.W.keys():
        stats_W[l].append(torch.linalg.norm(vae.W[l].weight.data.cpu(), ord = 2).numpy())
      cc=0
      for i,l in enumerate(vae.reconstruct):
        if type(l) == torch.nn.Linear:
          name = 'reconstruct' + str(cc)
          stats_reconstruct[name].append(torch.linalg.norm(l.weight.data.cpu(), ord = 2).numpy())
          cc+=1
  
      for i,(mu, logvar) in enumerate(zip(e_mu, e_logvar)):
        stats_e_mu['e'+str(i)].append(mu.mean().data.cpu().numpy())
        stats_e_logvar['e'+str(i)].append(logvar.mean().data.cpu().numpy())

      for i,(mu, logvar) in enumerate(zip(p_mu, p_logvar)):
        stats_p_mu['p'+str(i)].append(mu.mean().data.cpu().numpy())
        stats_p_logvar['p'+str(i)].append(logvar.mean().data.cpu().numpy())

      for i,(mu, logvar) in enumerate(zip(d_mu, d_logvar)):
        stats_d_mu['d'+str(i)].append(mu.mean().data.cpu().numpy())
        stats_d_logvar['d'+str(i)].append(logvar.mean().data.cpu().numpy())
  
  torch.save({
      'state_dict': vae.state_dict(), 
      'stats_encoder': stats_encoder,
      'stats_W': stats_W,
      'stats_reconstruct': stats_reconstruct,
      'stats_e_mu': stats_e_mu,
      'stats_e_logvar': stats_e_logvar,
      'stats_p_mu': stats_p_mu,
      'stats_p_logvar': stats_p_logvar,
      'stats_d_mu': stats_d_mu,
      'stats_d_logvar': stats_d_logvar,
      'stats':      stats,
      'args':       args,
  }, "trained.model.pth")

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
    local_dir='/zhome/45/c/127804/02460/Variational_proteins',
    resources_per_trial={'gpu':1})

print("Best config: ", analysis.get_best_config(
    metric="rho", mode="max"))
