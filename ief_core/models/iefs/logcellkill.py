import torch
import math
import numpy as np
import pickle 
from torch import nn
import torch.functional as F
from pyro.distributions import Normal, Independent, Categorical, LogNormal
import sys, os
from torch.autograd import grad

class LogCellKill(nn.Module):
    def __init__(self, dim_stochastic, dim_treat, dim_hidden = 300, mtype='logcellkill', response_only=False, alpha1_type='linear'):
        super(LogCellKill, self).__init__()
        self.dim_stochastic   = dim_stochastic
        self.dim_treat        = dim_treat
        self.mtype            = mtype
        self.rho              = nn.Parameter(torch.Tensor(dim_stochastic,))
        if alpha1_type == 'linear' or alpha1_type == 'quadratic' or alpha1_type == 'nl': 
            self.scale            = nn.Parameter(torch.Tensor(dim_stochastic,))
            self.controlfxn       = nn.Linear(dim_treat-1, dim_stochastic)
        elif alpha1_type == 'linear_fix': 
            self.scale        = nn.Parameter(torch.Tensor(1,))
            self.controlfxn   = nn.Linear(dim_treat-1, 1)
        # elif alpha1_type == 'nl': 
        #     self.scale      = nn.Parameter(torch.Tensor(dim_stochastic,))
        #     omodel          = nn.Sequential(nn.Linear(dim_treat-1, dim_hidden),nn.ReLU(True))
        #     self.controlfxn = nn.Sequential(omodel, nn.Linear(dim_hidden, dim_stochastic)) 
            
        self.response_only = response_only
        if not response_only:
            self.inpfxn       = nn.Linear(dim_stochastic, dim_stochastic)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.scale, 0.1)
        nn.init.constant_(self.rho, 0.5)
        
    def forward(self, inp, con):
        tvals = con[...,[0]]
        a     = con[...,1:]
        te    = torch.tanh(self.controlfxn(a))
        scale         = torch.sigmoid(self.scale)*2
        #self.debug   = [te, growth_term, scale, self.bias]
        if self.mtype == 'unused':
            # 73
            growth_term  = torch.sigmoid(self.rho)*torch.log(inp**2+1e-3)
            out          = inp*(1-growth_term-te*torch.exp(-tvals*scale))
            
            # 66
            growth_term  = torch.sigmoid(self.rho)*torch.log(inp**2+1e-3)
            out          = inp*(1-growth_term*0.-te*torch.exp(-tvals*scale)) 
            
            # 70
            growth_term  = torch.sigmoid(self.rho)*torch.nn.functional.softplus(inp)
            out          = inp*(1-growth_term-te*torch.exp(-tvals*scale))
        elif self.mtype=='logcellkill':
            growth_term  = torch.sigmoid(self.rho)*torch.log(inp**2+1e-3)
            out          = inp*(1-growth_term-te*torch.exp(-tvals*scale))
        elif self.mtype=='logcellkill_1':
            growth_term  = torch.sigmoid(self.rho)*torch.nn.functional.softplus(inp)
#             out          = inp*(1-growth_term*0.-te*torch.exp(-tvals*scale))
            out          = (1-growth_term*0.-te*torch.exp(-tvals*scale))
        else:
            growth_term  = torch.sigmoid(self.rho)*torch.nn.functional.softplus(inp)
            out          = inp*(1-growth_term-te*torch.exp(-tvals*scale)) 
        if self.response_only:
            return out
        else:
            return self.inpfxn(inp) + out