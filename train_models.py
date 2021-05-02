import torch 
import numpy as np
from misc import data, c
from models import *

from torch import optim
from scipy.stats import spearmanr
from torch.distributions.normal import Normal


class training(torch.nn.Module):
    def __init__(self, **kwargs):
      super(training, self).__init__()
      self.batch_size = kwargs['batch_size']
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.dataloader, self.df, self.mutants_tensor, self.mutants_df, self.weights, self.neff = data(batch_size = self.batch_size,neff_w = kwargs['neff_w'], device = device)

      self.wildtype   = self.dataloader.dataset[0] # one-hot-encoded wildtype 
      self.eval_batch = torch.cat([self.wildtype.unsqueeze(0), self.mutants_tensor])
      print('initialize train object')
      self.alphabet_size = self.dataloader.dataset[0].shape[0]
      self.seq_len = self.dataloader.dataset[0].shape[1]
      
    def train_HVAE(self,**kwargs):
      kwargs['alphabet_size'] = self.alphabet_size
      kwargs['seq_len'] = self.seq_len
      device = kwargs['device']
      
      step = kwargs['step']
      alpha_warm_up = torch.arange(0,1+step,step)
      vae   = HVAE(**kwargs).to(device)
      opt   = optim.Adam(vae.parameters(), lr = 0.000005)

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
      for i in range(len(kwargs['layers'])):
            stats_d_mu['d'+str(i)] = []
            stats_d_logvar['d'+str(i)] = []
            stats_e_mu['e'+str(i)] = []
            stats_e_logvar['e'+str(i)] = []
            stats_p_mu['p'+str(i)] = []
            stats_p_logvar['p'+str(i)] = []
            

      for epoch in range(kwargs['epochs']):
          # Unsupervised training on the MSA sequences.
          vae.train()
          if epoch > len(alpha_warm_up)-1:
            k = len(alpha_warm_up)-1
          else:
            k = epoch
          
          epoch_losses = { 'rl': [], 'kl': [] }
          for batch in self.dataloader:
              opt.zero_grad()
              x_hat,_, _,p_mu, p_logvar, d_mu, d_logvar = vae(batch)
              loss, rl, kl      = vae.loss(x_hat, batch,p_mu,p_logvar,d_mu,d_logvar,alpha_warm_up =alpha_warm_up[k])
              loss.mean().backward()
              opt.step()
              epoch_losses['rl'].append(rl.mean().item())
              epoch_losses['kl'].append(kl.mean().item())

          # Evaluation on mutants
          vae.eval()
          x_hat_eval, e_mu, e_logvar,p_mu, p_logvar, d_mu, d_logvar = vae(self.eval_batch)
          elbos, _, _ = vae.loss(x_hat_eval, self.eval_batch,p_mu,p_logvar,d_mu,d_logvar, alpha_warm_up =alpha_warm_up[k])#
          diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
          cor, _      = spearmanr(self.mutants_df.value, diffs.cpu().detach())
          
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
          'args':       kwargs,
      }, "trained.model_HVAE_l2.pth")


    def train_vanilla(self,**kwargs):
      device = kwargs['device']
      kwargs['alphabet_size'] = self.alphabet_size
      kwargs['seq_len'] = self.seq_len
      vae   = VAE(**kwargs).to(device)
      opt   = optim.Adam(vae.parameters())

      # rl  = Reconstruction loss
      # kl  = Kullback-Leibler divergence loss
      # cor = Spearman correlation to experimentally measured 
      #       protein fitness according to eq.1 from paper
      stats = { 'rl': [], 'kl': [], 'cor': [] }

      for epoch in range(kwargs['epoch']):
          # Unsupervised training on the MSA sequences.
          vae.train()
          
          epoch_losses = { 'rl': [], 'kl': [] }
          for batch in self.dataloader:
              opt.zero_grad()
              x_hat, mu, logvar = vae(batch)
              loss, rl, kl      = vae.loss(x_hat, batch, mu, logvar)
              loss.mean().backward()
              opt.step()
              epoch_losses['rl'].append(rl.mean().item())
              epoch_losses['kl'].append(kl.mean().item())

          # Evaluation on mutants
          vae.eval()
          x_hat_eval, mu, logvar = vae(self.eval_batch, rep=False)
          elbos, _, _ = vae.loss(x_hat_eval, self.eval_batch, mu, logvar)
          diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
          cor, _      = spearmanr(self.mutants_df.value, diffs.cpu().detach())
          
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

      torch.save({
          'state_dict': vae.state_dict(), 
          'stats':      stats,
          'args':       kwargs,
      }, "trained.model_vanilla.pth")

    def train_bayesian(self,**kwargs):
      
      device = kwargs['device']
      epochs = kwargs['epoch']
      bayesian = kwargs['bayesian']
      n_ensambles = kwargs['n_ensambles']
      kwargs['neff'] = self.neff.item()
      kwargs['alphabet_size'] = self.alphabet_size
      kwargs['seq_len'] = self.seq_len


      vae   = VAE_bayesian(**kwargs).to(device) #Initialize VAE model with the parameters stated above 
      opt   = optim.Adam(vae.parameters()) #Initialize Adam optimizer

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

          for batch in self.dataloader:
              opt.zero_grad()
              x_hat, mu, logvar = vae(batch) #Compute forward pass
              if bayesian:
                loss, rl, kl, KLB = vae.loss(x_hat, batch, mu, logvar) #Compute loss statistics
                loss.mean().backward() 
                opt.step()
                #Save statistics
                epoch_losses['rl'].append(rl.mean().item())
                epoch_losses['kl'].append(kl.mean().item())
                epoch_losses['KLB'].append(KLB.mean().item())

              else: #If not in bayesian setting no KLB is present
                loss, rl, kl = vae.loss(x_hat, batch, mu, logvar)
                loss.mean().backward()
                opt.step()
                epoch_losses['rl'].append(rl.mean().item())
                epoch_losses['kl'].append(kl.mean().item())
          
          # Evaluation on mutants
          vae.eval()
          with torch.no_grad(): #Ensure no gradients when computing the ensambles
            cor_lst = []

            # Ensemble over 256 iterations of the validation set
            if bayesian:
              if epoch % 8 == 0: #To speed computations up only do the ensambles every 8 epoch
                  mt_elbos, wt_elbos, ensambles = 0, 0, n_ensambles
                  for i in range(ensambles):
                      if i and (i % 2 == 0):
                          print(f"\tReached {i}", " "*32, end="\r")

                      elbos     = vae.logp_calc(self.eval_batch).detach().cpu() #Compute eblos needed for correlation computation
                      
                      #Split up computation as spearman correlation is not linear
                      wt_elbos += elbos[0]
                      mt_elbos += elbos[1:]
                  
                  print()

                  diffs = (mt_elbos / ensambles) - (wt_elbos / ensambles)
                  cor, _  = spearmanr(self.mutants_df.value, diffs) #Compute the correlation

            else: #If not doing bayesian no need for ensambles
              elbos       = vae.logp_calc(self.eval_batch)
              diffs       = elbos[1:] - elbos[0] # log-ratio (first equation in the paper)
              cor, _      = spearmanr(self.mutants_df.value, diffs.cpu().detach())

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
            'args':       kwargs,
        }, "trained.model.bayesian_l2.pth")
      

if __name__ == '__main__':
  
  train_type = 'Bayesian'
  kwargs_vae =  {
            'epoch': 32,
            'device':torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'train_type': train_type
        }
  kwargs_bayesian = {
            'epoch': 300,
            'n_ensambles': 256,
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'bayesian': True,
            'beta': 1,
            'hidden_size': 2000,
            'latent_size': 30,
            'shared_size': 40,
            'repeat': 1,
            'group_sparsity': True,
            'dropout': 0,
            'train_type': train_type

        }
  kwargs_hvae = {
        'device':torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'step': 0.01,
        'hidden_size': 1000,
        'layers': [256,512,512],
        'latents': [8,8,16],
        'epochs': 50,
        'train_type': train_type

    }



  kwargs_init = {'batch_size': 128, 'neff_w': True}
  train = training(**kwargs_init)

  if train_type == 'Vanilla':
    train.train_vanilla(**kwargs_vae)
  elif train_type == 'Bayesian':
    train.train_bayesian(**kwargs_bayesian)
  elif train_type == 'HVAE':
    train.train_HVAE(**kwargs_hvae)
