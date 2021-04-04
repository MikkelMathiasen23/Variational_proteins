import torch
import numpy as np
from misc import data, c
from torch import optim
from scipy.stats import spearmanr
from torch.distributions.normal import Normal
from vae import VAE_bayesian

def train_vae(epochs=200,n = 256, batch_size =128,bayesian= True, group_sparsity=True, beta=1, shared_size= 40, dropout = 0, repeat = 1):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dataloader, df, mutants_tensor, mutants_df, weights, neff = data(batch_size = batch_size, device = device)

  wildtype   = dataloader.dataset[0] # one-hot-encoded wildtype 
  eval_batch = torch.cat([wildtype.unsqueeze(0), mutants_tensor])
  args = {
      'alphabet_size': dataloader.dataset[0].shape[0],
      'seq_len':       dataloader.dataset[0].shape[1],
      'neff': neff.item(),
      'device': device,
      'bayesian': bayesian,
      'beta': beta,
      'hidden_size': 2000,
      'latent_size': 30,
      'shared_size': shared_size,
      'repeat': repeat,
      'group_sparsity': group_sparsity,
      'dropout': dropout
  }

  vae   = VAE_bayesian(**args).to(device)
  opt   = optim.Adam(vae.parameters())

  # rl  = Reconstruction loss
  # kl  = Kullback-Leibler divergence loss
  # cor = Spearman correlation to experimentally measured 
  #       protein fitness according to eq.1 from paper
  if bayesian:
    stats = { 'rl': [], 'kl': [], 'cor': [],'KLB': []}
  else:
    stats = { 'rl': [], 'kl': [], 'cor': []}

  for epoch in range(epochs):
      # Unsupervised training on the MSA sequences.
      vae.train()
      if bayesian:
        epoch_losses = { 'rl': [], 'kl': [], 'cor': [],'KLB': []}
      else:
        epoch_losses = { 'rl': [], 'kl': [], 'cor': []}

      for batch in dataloader:
          opt.zero_grad()
          x_hat, mu, logvar = vae(batch)
          if bayesian:
            loss, rl, kl, KLB = vae.loss(x_hat, batch, mu, logvar)
            loss.mean().backward()
            opt.step()
            epoch_losses['rl'].append(rl.mean().item())
            epoch_losses['kl'].append(kl.mean().item())
            epoch_losses['KLB'].append(KLB.mean().item())

          else: 
            loss, rl, kl = vae.loss(x_hat, batch, mu, logvar)
            loss.mean().backward()
            opt.step()
            epoch_losses['rl'].append(rl.mean().item())
            epoch_losses['kl'].append(kl.mean().item())
      
      # Evaluation on mutants
      vae.eval()
      with torch.no_grad():
        cor_lst = []

        # Ensemble over 256 iterations of the validation set
        if bayesian:
          if epoch % 8 == 0:
              mt_elbos, wt_elbos, ensambles = 0, 0, n
              for i in range(ensambles):
                  if i and (i % 2 == 0):
                      print(f"\tReached {i}", " "*32, end="\r")

                  elbos     = vae.logp_calc(eval_batch).detach().cpu()
                  wt_elbos += elbos[0]
                  mt_elbos += elbos[1:]
              
              print()

              diffs = (mt_elbos / ensambles) - (wt_elbos / ensambles)
              cor, _  = spearmanr(mutants_df.value, diffs)

        else:
          elbos       = vae.logp_calc(eval_batch)
          diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
          cor, _      = spearmanr(mutants_df.value, diffs.cpu().detach())

        # Populate statistics 
        stats['rl'].append(np.mean(epoch_losses['rl']))
        stats['kl'].append(np.mean(epoch_losses['kl']))
        stats['cor'].append(np.abs(cor))

        if bayesian:
          stats['KLB'].append(np.mean(epoch_losses['KLB']))

      if bayesian:
          to_print = [
            f"{c.HEADER}EPOCH %03d"          % epoch,
            f"{c.OKBLUE}RL=%4.4f"            % stats['rl'][-1], 
            f"{c.OKGREEN}KL=%4.4f"           % stats['kl'][-1],
            f"{c.OKGREEN}KLB=%4.4f"           % stats['KLB'][-1], 
            f"{c.OKCYAN}|rho|=%4.4f{c.ENDC}" % stats['cor'][-1]
        ]
      
      else:
        to_print = [
            f"{c.HEADER}EPOCH %03d"          % epoch,
            f"{c.OKBLUE}RL=%4.4f"            % stats['rl'][-1], 
            f"{c.OKGREEN}KL=%4.4f"           % stats['kl'][-1], 
            f"{c.OKCYAN}|rho|=%4.4f{c.ENDC}" % stats['cor'][-1]
        ]
      print(" ".join(to_print))

  if bayesian:
    torch.save({
        'state_dict': vae.state_dict(), 
        'stats':      stats,
        'args':       args,
    }, "trained.model.bayesian.pth")
  else:
    torch.save({
        'state_dict': vae.state_dict(), 
        'stats':      stats,
        'args':       args,
    }, "trained.model.non_bayesian.pth")
