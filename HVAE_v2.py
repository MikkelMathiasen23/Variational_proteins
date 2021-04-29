import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

class HVAE(torch.nn.Module):

    def __init__(self, **kwargs):     
        super(HVAE, self).__init__()

        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len       = kwargs['seq_len']
        self.input_size    = self.alphabet_size * self.seq_len
        self.device        = kwargs['device']
        self.shared_size   = kwargs['shared_size']
        self.repeat        = kwargs['repeat']
        self.group_sparsity = kwargs['group_sparsity']
        self.L              = kwargs['layers']
        self.latents        = kwargs['latents']
        self.hidden_size   = kwargs['hidden_size']

        
        self.ll = len(self.L)
        self.e = torch.nn.ModuleDict({})
        self.W = torch.nn.ModuleDict({})
        for i,l in enumerate(self.L):
          name = 'e'+str(i)
          if i==0:
            self.e[name] = torch.nn.Linear(self.input_size, l).to(self.device)
            self.W[name] = torch.nn.Linear(l,self.latents[i]).to(self.device)
          else:
            self.e[name] = torch.nn.Linear(self.L[i-1], l).to(self.device)
            self.W[name] = torch.nn.Linear(l,self.latents[i]).to(self.device)

          torch.nn.init.xavier_normal_(self.e[name].weight)
          torch.nn.init.xavier_normal_(self.W[name].weight)
          torch.nn.init.constant_(self.e[name].bias, -5)
          torch.nn.init.constant_(self.W[name].bias, -5)

        torch.nn.init.constant_(self.e['e0'].bias, -20)
        torch.nn.init.constant_(self.W['e0'].bias, -20)

        self.upscale = torch.nn.ModuleDict({})
        for l in range(len(self.L)-1,-1,-1):
          name = 'd'+str(l)
          if l == 0:
            self.W[name] = torch.nn.Linear(self.latents[l],self.latents[l] ).to(self.device)
          else:
            self.W[name] = torch.nn.Linear(self.latents[l], self.latents[l-1]).to(self.device)
            if self.latents[l] != self.latents[l-1]:
              self.upscale[name] = torch.nn.Linear(self.latents[l], self.latents[l-1]).to(self.device)
              torch.nn.init.xavier_normal_(self.upscale[name].weight)
              torch.nn.init.constant_(self.upscale[name].bias, -5)
          torch.nn.init.xavier_normal_(self.W[name].weight)
          torch.nn.init.constant_(self.W[name].bias, -10)

        if self.group_sparsity:
          self.reconstruct = torch.nn.Sequential(
              torch.nn.Linear(self.latents[0], 100),
              torch.nn.LeakyReLU(),
              torch.nn.Linear(100, self.hidden_size // 4),
              torch.nn.Sigmoid()        
          )
          #Define linear layers needed for creating group saprsity of last linear layer of the decoder
          self.W_g = torch.nn.Linear(self.shared_size, (self.hidden_size//4) * self.seq_len, bias = False ) #Reduce the computation of D and S by down-scaling the size 
          self.D = torch.nn.Linear(self.alphabet_size,self.shared_size , bias = False)
          self.S = torch.nn.Linear(self.seq_len, (self.hidden_size//4)//self.repeat, bias = False) #Down-scale as with W, but can also imply that there only are needed 
          #If not in bayesian setting then initialize without mean and logvar:
          self.lambda_  = torch.nn.Parameter(torch.Tensor([0.1] * self.input_size))
          self.b_       = torch.nn.Parameter(torch.Tensor([0.1] * self.input_size))
        else:
          self.reconstruct = torch.nn.Sequential(
                                      torch.nn.Linear(self.latents[0],100),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(100, self.hidden_size),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(self.hidden_size,self.input_size))
         
        for l in self.reconstruct:
          if type(l) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(l.weight)
            torch.nn.init.constant_(l.bias, -5)           

        self.relu = torch.nn.ReLU()
        self.leakyrelu = torch.nn.LeakyReLU()
        self.softplus = torch.nn.Softplus()
        

    def sampler(self, mu,logvar):
      return mu + torch.randn_like(mu)*(0.5*logvar).exp()

    def ed_sampler(self, h, name):
        mu = self.W[name](h)
        logvar = self.softplus(mu)

        return self.sampler(mu,logvar), mu, logvar

    def eq_21(self,mu1,mu2, sigma1,sigma2, name):

        if mu1.shape[1] != mu2.shape[1]:
          mu1 = self.upscale[name](mu1)
          sigma1 = self.upscale[name](sigma1)

        sigma1_sq = 1 / sigma1.pow(2)
        sigma2_sq = 1 / sigma2.pow(2)
        
        mu = (mu1*sigma1_sq + mu2*sigma2_sq)/(sigma1_sq + sigma2_sq)
        sigma = 1 / (sigma1_sq + sigma2_sq)
        logsigma = torch.log(sigma + 1e-8)


        return self.sampler(mu, logsigma),mu, logsigma

    def encoder(self, x):
        h = x
        e_mu = [0]*self.ll
        e_logvar = [0]*self.ll
        for i in range(len(self.L)):
          name = 'e'+str(i)
          h = self.e[name](h)
          h = self.leakyrelu(h)
          
          _, e_mu[i], e_logvar[i] = self.ed_sampler(h, name)  
          
        return e_mu, e_logvar

    def decoder(self, e_mu, e_logvar):
      p_mu, p_logvar = [0]*self.ll, [0]*self.ll
      d_mu, d_logvar = [0]*self.ll, [0]*self.ll
      for i in range(len(self.L)-1, -1, -1):
        
        name = 'd' + str(i)
        if i == len(self.L)-1:
          d_mu[i], d_logvar[i] = e_mu[i], e_logvar[i]

          z = self.sampler(d_mu[i], d_logvar[i])

          p_mu[i] = (torch.zeros_like(d_mu[i]).to(self.device))
          p_logvar[i] = (torch.eye(d_logvar[i].shape[0],d_logvar[i].shape[1]).to(self.device))
        else:
          name = 'd' + str(i+1)
          _, p_mu[i], p_logvar[i] = self.ed_sampler(z, name)

          z, d_mu[i], d_logvar[i] = self.eq_21(e_mu[i], p_mu[i], e_logvar[i].mul(1/2).exp(), p_logvar[i].mul(1/2).exp(),name)

      x = self.reconstruct(z)
      if self.group_sparsity:
        lambda_ = self.lambda_
        b_      = self.b_
        
        #Reshape S into the full shape and pass through sigmoid layer to drive values towards 0 or 1
        S = torch.sigmoid(self.S.weight.repeat(self.repeat,1))
        S = S.unsqueeze(-2)

        #Multiply W and D:
        W_scale = torch.mm(self.W_g.weight,self.D.weight)
        #Reshape W into the needed shape:
        W_scale = W_scale.view(self.hidden_size//4, self.alphabet_size, self.seq_len)
        #Element-wise multiplication of W and S:
        W_scale = (W_scale*S).view(-1, self.input_size)

        #Calculate the final output from the linear layer using the calculation from the article:
        x = (1 + lambda_.exp()).log() *F.linear(x, W_scale.T, bias = b_)

      return x, p_mu, p_logvar, d_mu, d_logvar


    def forward(self, x):
      x = x.view(-1, self.input_size)        # flatten

      e_mu, e_logvar = self.encoder(x)
      x,p_mu, p_logvar, d_mu, d_logvar = self.decoder(e_mu, e_logvar)

      x = x.view(-1, self.alphabet_size, self.seq_len)        
      x = x.log_softmax(dim=1)                           # softmax
      return x, e_mu, e_logvar,p_mu, p_logvar, d_mu, d_logvar

    def loss(self, x_hat, true_x, p_mu, p_logvar, d_mu, d_logvar, alpha_warm_up ):
      RL = -(x_hat*true_x).sum(-1).sum(-1)
      KL = []
      
      for i in range(len(p_mu)):
        p = Normal(p_mu[i], p_logvar[i].mul(1/2).exp())
        q = Normal(d_mu[i], d_logvar[i].mul(1/2).exp())
        KL.append(kl_divergence(p,q).sum(-1))
      loss = (RL + 1/self.ll*torch.stack(KL, dim = 1).sum(-1)*alpha_warm_up)
      #Try to scale this with 1/n.latents*beta

      return loss, RL, 1/self.ll*torch.stack(KL, dim = 1).sum(-1)*alpha_warm_up