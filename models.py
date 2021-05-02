import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
import numpy as np


class VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.hidden_size   = 64
        self.latent_size   = 2
        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len       = kwargs['seq_len']
        self.input_size    = self.alphabet_size * self.seq_len

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )

        # Latent space `mu` and `var`
        self.fc21 = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = torch.nn.Linear(self.hidden_size, self.latent_size)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.input_size),
        )

    def forward(self, x, rep=True):
        x = x.view(-1, self.input_size)                    # flatten
        x = self.encoder(x)                                # encode
        mu, logvar = self.fc21(x), self.fc22(x)            # branch mu, var

        if rep:                                            # reparameterize
            x = mu + torch.randn_like(mu) * (0.5*logvar).exp() 
        else:                                              # or don't 
            x = mu                                         

        x = self.decoder(x)                                # decode
        x = x.view(-1, self.alphabet_size, self.seq_len)   # squeeze back
        x = x.log_softmax(dim=1)                           # softmax
        return x, mu, logvar
    
    def loss(self, x_hat, true_x, mu, logvar, beta=0.5):
        RL = -(x_hat*true_x).sum(-1).sum(-1)                    # reconst. loss
        KL = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1) # KL loss
        return RL + beta*KL, RL, KL





class VAE_bayesian(torch.nn.Module):
    def __init__(self, **kwargs):     
        super(VAE_bayesian, self).__init__()

        self.neff          = kwargs['neff']
        self.hidden_size   = kwargs['hidden_size']
        self.latent_size   = kwargs['latent_size']
        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len       = kwargs['seq_len']
        self.input_size    = self.alphabet_size * self.seq_len
        self.device        = kwargs['device']
        self.bayesian      = kwargs['bayesian']
        self.beta          = kwargs['beta']
        self.shared_size   = kwargs['shared_size']
        self.repeat        = kwargs['repeat']
        self.group_sparsity = kwargs['group_sparsity']
        self.dropout       = kwargs['dropout']


        def bayesian_register_weights(l,name=None):
          ##Helper function to initialize weights and bias
          ##Register mean and logvar parameters for weights and bias
          ##Register forward pre-hooks to sample weights and bias before forward pass

          #Delete weight attribute of layer as a new weight mean and logvar are created:
          del l._parameters['weight']
          #If the layer contains a bias remove it:
          if l.bias is not None:
            contains_bias = True
            del l._parameters['bias']
          else: 
            contains_bias = False #Otherwise flag that there are no bias
          setattr(l, "name", name)

          #Set-up mean and logvar parameters for weights and register these:
          weight_mean_param   = torch.nn.Parameter(torch.Tensor(l.out_features, l.in_features))
          weight_logvar_param = torch.nn.Parameter(torch.Tensor(l.out_features, l.in_features))
          l.register_parameter('weight_mean', weight_mean_param)
          l.register_parameter('weight_logvar', weight_logvar_param)

          #Initialize weight mean using xavier normal:
          var = 2/(l.out_features + l.in_features)
          torch.nn.init.normal_(weight_mean_param, 0.0, std = var**(1/2))
          #torch.nn.init.xavier_normal_(weight_mean_param)
          #Initialize logvar using constant:
          torch.nn.init.constant_(weight_logvar_param, -5)

          if contains_bias:
            #If layer has bias initialize bias mean and logvar as well and register parameters:
            bias_mean_param = torch.nn.Parameter(torch.Tensor(l.out_features))
            bias_logvar_param = torch.nn.Parameter(torch.Tensor(l.out_features))    
            l.register_parameter('bias_mean',bias_mean_param)
            l.register_parameter('bias_logvar',bias_logvar_param)
            torch.nn.init.constant_(bias_mean_param,  0.1)
            torch.nn.init.constant_(bias_logvar_param, -10)

          #Register pre-hook to be performed before forward pass:
          l.register_forward_pre_hook(self.hook_weight)
          self.hook_weight(l,None)
          if contains_bias:
            #If layer has bias register pre-hook as well:
            l.register_forward_pre_hook(self.hook_bias)
            self.hook_bias(l,None)

        #######################################################################
        ##Define layers: 
        #######################################################################
        
        #Define decoder architecture:
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1500),
            torch.nn.ReLU(),
            torch.nn.Linear(1500, 1500),
            torch.nn.ReLU()
        )
        #Define latent space mean and logvar:
        self.fc21 = torch.nn.Linear(1500, self.latent_size)
        self.fc22 = torch.nn.Linear(1500, self.latent_size)
        
        #Initialize encoder weights and bias:
        for l in self.encoder:
            if type(l) == torch.nn.Linear:
              torch.nn.init.xavier_normal_(l.weight)
              torch.nn.init.constant_(l.bias, 0.1)

        #Initialize latent space mean and logvar:        
        torch.nn.init.constant_(self.fc22.bias, -5)
        torch.nn.init.constant_(self.fc21.bias, 0.1)
        torch.nn.init.xavier_normal_(self.fc21.weight)
        torch.nn.init.xavier_normal_(self.fc22.weight)

        #Define decoder architecture - depends if group-sparsity or not.
        #If group-sparsity the last linear layer is computed according to equation in article.
        if self.group_sparsity:
          self.decoder = torch.nn.Sequential(
              torch.nn.Linear(self.latent_size, 125),
              torch.nn.Dropout(p = self.dropout),
              torch.nn.ReLU(),
              torch.nn.Linear(125, self.hidden_size // 4),
              torch.nn.Dropout(p = self.dropout),
              torch.nn.Sigmoid()        
          )
          #Define linear layers needed for creating group saprsity of last linear layer of the decoder
          self.W = torch.nn.Linear(self.shared_size, (self.hidden_size//4) * self.seq_len, bias = False ) #Reduce the computation of D and S by down-scaling the size 
          self.D = torch.nn.Linear(self.alphabet_size,self.shared_size , bias = False)
          self.S = torch.nn.Linear(self.seq_len, (self.hidden_size//4)//self.repeat, bias = False) #Down-scale as with W, but can also imply that there only are needed 
                                                                                                   #to learn fewer parameters defined by self.repeat             
        else:
            self.decoder = torch.nn.Sequential(
              torch.nn.Linear(self.latent_size, 100),
              torch.nn.Dropout(p = self.dropout),
              torch.nn.ReLU(),
              torch.nn.Linear(100, self.hidden_size),
              torch.nn.Dropout(p = self.dropout),
              torch.nn.Sigmoid(),
              torch.nn.Linear(self.hidden_size, self.input_size)
          )

        #If bayesian modelling:
        if self.bayesian:
          self.sample_layers = []
          #Initialize weights and biases, and register pre-hook:
          for l in self.decoder:
            if type(l) == torch.nn.Linear:
              bayesian_register_weights(l, name = None)
              self.sample_layers.append(l)
          if self.group_sparsity:
            #If model contains group sparsity register these weights but without any bias:
            bayesian_register_weights(self.W, name = None)
            bayesian_register_weights(self.D, name = None)
            bayesian_register_weights(self.S, name = 'S')
            self.sample_layers.extend([self.W, self.D, self.S])

            #Register and initialize lambda parameter needed for group-sparsity computation:
            self.lambda_mean   = torch.nn.Parameter(torch.Tensor([1]))
            self.lambda_logvar = torch.nn.Parameter(torch.Tensor([-5]))

            #Register and initialize bias parameter for the last layer of the decoder implying group-sparsity:
            self.b_mean   = torch.nn.Parameter(torch.Tensor([0.1] * self.input_size))
            self.b_logvar = torch.nn.Parameter(torch.Tensor([-5] * self.input_size))
        
        elif self.group_sparsity:
          #If not in bayesian setting then initialize without mean and logvar:
          self.lambda_  = torch.nn.Parameter(torch.Tensor([0.1] * self.input_size))
          self.b_       = torch.nn.Parameter(torch.Tensor([0.1] * self.input_size))
             
    #Hook functions needed for sampling weights and bias before forward pass:
    def hook_weight(self,l,input):
        setattr(l, 'weight',Normal(l.weight_mean, l.weight_logvar.mul(1/2).exp()).rsample())
    def hook_bias(self,l,input):
        setattr(l, 'bias',Normal(l.bias_mean, l.bias_logvar.mul(1/2).exp()).rsample())

    #Sample function for latent space:
    def sampler(self, mu, logvar):
        return mu + torch.randn_like(mu)*(0.5*logvar).exp()
    
    def decode_sparse(self,x):
      #Function to perform group-sparsity calculations for the last linear layer of the decoder

      if self.bayesian: #If bayesian setting then it is needed to sample the weights of the 
                        #matrices needed for group sparsity. This is needed as we do not perform
                        #forward pass directly on these layers
          self.hook_weight(self.D, None) #Sample for D, W and S
          self.hook_weight(self.W, None)
          self.hook_weight(self.S, None)
          #Sample weights using mean and logvar for lambda and bias for this layer:
          lambda_ = Normal(self.lambda_mean, self.lambda_logvar.mul(1/2).exp()).rsample()
          b_      = Normal(self.b_mean, self.b_logvar.mul(1/2).exp()).rsample()
      else: 
        #If not bayesian then just use the values:
        lambda_ = self.lambda_
        b_      = self.b_
      
      #Reshape S into the full shape and pass through sigmoid layer to drive values towards 0 or 1
      S = torch.sigmoid(self.S.weight.repeat(self.repeat,1))
      S = S.unsqueeze(-2)

      #Multiply W and D:
      W_scale = torch.mm(self.W.weight,self.D.weight)
      #Reshape W into the needed shape:
      W_scale = W_scale.view(self.hidden_size//4, self.alphabet_size, self.seq_len)
      #Element-wise multiplication of W and S:
      W_scale = (W_scale*S).view(-1, self.input_size)

      #Calculate the final output from the linear layer using the calculation from the article:
      return (1 + lambda_.exp()).log() *F.linear(x, W_scale.T, bias = b_)

    
    def forward(self, x):
        x = x.view(-1, self.input_size)        # flatten
        
        # Encoding
        x = self.encoder(x)

        #Latent layer mean and logvar and sample using this:
        mu_latent, logvar_latent = self.fc21(x), self.fc22(x)
        x = self.sampler(mu_latent, logvar_latent)

        # Decoder
        x = self.decoder(x)

        if self.group_sparsity:
          #Perform group sparsity:
          x = self.decode_sparse(x)
        
        #Reshape to output shape:
        x = x.view(-1, self.alphabet_size, self.seq_len)        
        x = x.log_softmax(dim=1)                           # softmax
        return x, mu_latent, logvar_latent

    def loss(self, x_hat, true_x, mu, logvar):
        
        # Compute reconstruction loss:
        RL = -(x_hat*true_x).sum(-1).sum(-1)
        #Compute KL divergence term for the latent space:
        kl_latent = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1)

        #Calculate the loss of RL + KL of latent:
        out = (RL + self.beta*kl_latent).mean() #Can use the beta term to weight the kl-latent

        if self.bayesian: #Compute KL divergence term for all bayesian parameters including bias and weights.

          KLB = torch.tensor([0]).float().to(self.device) #Initialize as zero and then just add the coming values:
          for l in self.sample_layers:
            if type(l) == torch.nn.Linear:
                #Compute normal distribution using the parameters for the layer:
                mu, logvar  =   l.weight_mean, l.weight_logvar
                dist_       =   Normal(mu, logvar.mul(1/2).exp())

                if False:#l.name == 'S':
                   #S matrix is a special case as we want another prior distribution N(-9.3053, log(4))
                  dist_prior_ =   Normal(-9.3053*torch.ones_like(mu), np.log(4)*torch.ones_like(logvar))                
                else:
                   #For the rest we want prior N(0,1):
                  dist_prior_ =   Normal(torch.zeros_like(mu), torch.ones_like(logvar))
                KLB        +=   kl_divergence(dist_, dist_prior_).sum() #Compute convergence
                if l.bias is not None: #Do the same for bias if the layer has that
                  #Bias:
                  bias_mu, bias_logvar =  l.bias_mean, l.bias_logvar
                  dist_       = Normal(bias_mu, bias_logvar.mul(1/2).exp())
                  dist_prior_ = Normal(torch.zeros_like(bias_mu), torch.ones_like(bias_logvar))
                  KLB        += kl_divergence(dist_, dist_prior_).sum()
          
          if self.group_sparsity: #It is also needed to compute the KL divergence for lambda and bias of the group-sparsity output layer
            #       lambda:          
            dist_lambda       = Normal(self.lambda_mean, self.lambda_logvar.mul(1/2).exp())
            dist_prior_lambda = Normal(torch.zeros_like(self.lambda_mean), torch.ones_like(self.lambda_logvar))
            KLB += kl_divergence(dist_lambda, dist_prior_lambda).sum()

            #       Weight out bias:
            dist_b       = Normal(self.b_mean, self.b_logvar.mul(1/2).exp())
            dist_prior_b = Normal(torch.zeros_like(self.b_mean), torch.ones_like(self.b_logvar))
            KLB += kl_divergence(dist_b, dist_prior_b).sum()

          KLB/=self.neff #Divide the KL divergence term with Neff from sequence weighting 
          #Total loss including RL + KL_latent and KLB
          out = out + KLB
          return out  , RL, self.beta*kl_latent, KLB


        return out, RL, self.beta*kl_latent

    def logp_calc(self, x):
        #Helper function used to calculate the needed KL of the latent space and reconstruction loss
        #to perform the correlation coefficient

        #Perform forward pass with sampling of weights:
        x_hat, mu, logvar = self(x)
        
        #Calculate reconstruction loss:
        RL = -(x_hat*x).sum(-1).sum(-1)
        #Calculate KL-divergence of the latent space:
        kl_latent = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1)

        return RL + self.beta*kl_latent



class HVAE(torch.nn.Module):

    def __init__(self, **kwargs):     
        super(HVAE, self).__init__()

        self.alphabet_size = kwargs['alphabet_size']
        self.seq_len       = kwargs['seq_len']
        self.input_size    = self.alphabet_size * self.seq_len
        self.device        = kwargs['device']
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

      return loss, RL, 1/self.ll*torch.stack(KL, dim = 1).sum(-1)*alpha_warm_up
